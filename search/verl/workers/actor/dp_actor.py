# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_micro_batch_with_logits(
        self, micro_batch, temperature,
        kl_topk_k: int = None, kl_topk_indices: torch.Tensor = None,
    ) -> tuple:
        """Forward pass that also returns logits for KL divergence computation.

        Args:
            micro_batch: Input batch dict.
            temperature: Temperature for logits.
            kl_topk_k: -1 for full logits, >0 to compute top-k from own logits.
                       None means use kl_topk_indices instead.
            kl_topk_indices: Indices to gather at (only used when kl_topk_k is None).

        Returns:
            entropy: (bs, response_len)
            log_probs: (bs, response_len)
            kl_inputs: dict containing:
                - logits_k: (bs, response_len, vocab_size) if kl_topk_k=-1, else (bs, response_len, k)
                - topk_indices: (bs, response_len, k) or None if kl_topk_k=-1
                - logsumexp: (bs, response_len) - for proper normalization
        """
        if kl_topk_k is None and kl_topk_indices is None:
            raise ValueError("Must provide either kl_topk_k or kl_topk_indices")
        if kl_topk_k is not None and kl_topk_indices is not None:
            raise ValueError("kl_topk_k and kl_topk_indices are mutually exclusive")

        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)

                # Compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                # Compute log probs
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # Derive logsumexp from log_probs: logsumexp = logits[label] - log_softmax(logits)[label]
                logits_at_label_rmpad = logits_rmpad.gather(-1, input_ids_rmpad_rolled.unsqueeze(-1)).squeeze(-1)
                logsumexp_rmpad = logits_at_label_rmpad - log_probs

                # Handle top-k or full logits
                if kl_topk_k == -1:
                    logits_k_rmpad = logits_rmpad
                    topk_indices_rmpad = None
                elif kl_topk_k is not None and kl_topk_k > 0:
                    _, topk_indices_rmpad = logits_rmpad.topk(kl_topk_k, dim=-1)
                    logits_k_rmpad = logits_rmpad.gather(-1, topk_indices_rmpad)
                else:
                    # kl_topk_k is None, use kl_topk_indices - gather later
                    logits_k_rmpad = None
                    topk_indices_rmpad = None

                # Gather if sp > 1
                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if logits_k_rmpad is not None:
                        logits_k_rmpad = gather_outpus_and_unpad(logits_k_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    else:
                        logits_rmpad = gather_outpus_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    logsumexp_rmpad = gather_outpus_and_unpad(logsumexp_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if topk_indices_rmpad is not None:
                        topk_indices_rmpad = gather_outpus_and_unpad(topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # Handle kl_topk_indices case: gather directly at response positions
                if kl_topk_k is None and kl_topk_indices is not None:
                    # Validate kl_topk_indices shape
                    assert kl_topk_indices.shape[0] == batch_size, \
                        f"kl_topk_indices batch size {kl_topk_indices.shape[0]} != expected {batch_size}"
                    assert kl_topk_indices.shape[1] == response_length, \
                        f"kl_topk_indices response_length {kl_topk_indices.shape[1]} != expected {response_length}"

                    total_nnz = logsumexp_rmpad.shape[0]
                    inverse_indices = torch.full((batch_size * seqlen,), -1, dtype=torch.long, device=logsumexp_rmpad.device)
                    inverse_indices[indices] = torch.arange(total_nnz, device=logsumexp_rmpad.device)

                    response_start = seqlen - response_length - 1
                    batch_offsets = torch.arange(batch_size, device=logsumexp_rmpad.device) * seqlen
                    response_offsets = torch.arange(response_length, device=logsumexp_rmpad.device)
                    flattened_response_pos = batch_offsets.unsqueeze(1) + response_start + response_offsets.unsqueeze(0)
                    rmpad_response_pos = inverse_indices[flattened_response_pos]
                    valid_mask = rmpad_response_pos >= 0
                    safe_rmpad_pos = rmpad_response_pos.clamp(min=0)

                    k = kl_topk_indices.shape[-1]
                    rmpad_pos_expanded = safe_rmpad_pos.unsqueeze(-1).expand(-1, -1, k)
                    logits_k = logits_rmpad[rmpad_pos_expanded.reshape(-1), kl_topk_indices.reshape(-1)]
                    logits_k = logits_k.reshape(batch_size, response_length, k)
                    logsumexp = logsumexp_rmpad[safe_rmpad_pos]
                    logits_k = logits_k * valid_mask.unsqueeze(-1)
                    logsumexp = logsumexp * valid_mask

                    kl_inputs = {"logits_k": logits_k, "topk_indices": kl_topk_indices, "logsumexp": logsumexp}
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                    full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
                else:
                    # Standard path: pad back to (bsz, seqlen), then slice
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logits_k = pad_input(hidden_states=logits_k_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logsumexp = pad_input(hidden_states=logsumexp_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    if topk_indices_rmpad is not None:
                        full_topk_indices = pad_input(hidden_states=topk_indices_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)

                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
                    logits_k = full_logits_k[:, -response_length - 1:-1, :]
                    logsumexp = full_logsumexp.squeeze(-1)[:, -response_length - 1:-1]

                    if kl_topk_k == -1:
                        kl_inputs = {"logits_k": logits_k, "topk_indices": None, "logsumexp": logsumexp}
                    else:
                        topk_indices = full_topk_indices[:, -response_length - 1:-1, :]
                        kl_inputs = {"logits_k": logits_k, "topk_indices": topk_indices, "logsumexp": logsumexp}

                # Shape assertions for rmpad branch
                batch_size_out = log_probs.shape[0]
                seq_len_out = log_probs.shape[1]
                assert kl_inputs["logsumexp"].shape == (batch_size_out, seq_len_out), \
                    f"logsumexp shape {kl_inputs['logsumexp'].shape} != expected ({batch_size_out}, {seq_len_out})"
                assert kl_inputs["logits_k"].shape[0] == batch_size_out and kl_inputs["logits_k"].shape[1] == seq_len_out, \
                    f"logits_k shape {kl_inputs['logits_k'].shape} batch/seq mismatch with log_probs {log_probs.shape}"
                if kl_inputs["topk_indices"] is not None:
                    assert kl_inputs["topk_indices"].shape == kl_inputs["logits_k"].shape, \
                        f"topk_indices shape {kl_inputs['topk_indices'].shape} != logits_k shape {kl_inputs['logits_k'].shape}"

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)

                # Derive logsumexp from log_probs
                logits_at_label = logits.gather(-1, micro_batch['responses'].unsqueeze(-1)).squeeze(-1)
                logsumexp = logits_at_label - log_probs

                if kl_topk_k == -1:
                    kl_inputs = {"logits_k": logits, "topk_indices": None, "logsumexp": logsumexp}
                elif kl_topk_k is not None and kl_topk_k > 0:
                    _, topk_indices = logits.topk(kl_topk_k, dim=-1)
                    logits_k = logits.gather(-1, topk_indices)
                    kl_inputs = {"logits_k": logits_k, "topk_indices": topk_indices, "logsumexp": logsumexp}
                else:
                    # kl_topk_k is None, gather at provided kl_topk_indices
                    assert kl_topk_indices is not None
                    # Validate kl_topk_indices shape
                    assert kl_topk_indices.shape[0] == batch_size, \
                        f"kl_topk_indices batch size {kl_topk_indices.shape[0]} != expected {batch_size}"
                    assert kl_topk_indices.shape[1] == response_length, \
                        f"kl_topk_indices response_length {kl_topk_indices.shape[1]} != expected {response_length}"
                    logits_k = logits.gather(-1, kl_topk_indices)
                    kl_inputs = {"logits_k": logits_k, "topk_indices": kl_topk_indices, "logsumexp": logsumexp}

                # Shape assertions for non-rmpad branch
                batch_size_out = log_probs.shape[0]
                seq_len_out = log_probs.shape[1]
                assert kl_inputs["logsumexp"].shape == (batch_size_out, seq_len_out), \
                    f"logsumexp shape {kl_inputs['logsumexp'].shape} != expected ({batch_size_out}, {seq_len_out})"
                assert kl_inputs["logits_k"].shape[0] == batch_size_out and kl_inputs["logits_k"].shape[1] == seq_len_out, \
                    f"logits_k shape {kl_inputs['logits_k'].shape} batch/seq mismatch with log_probs {log_probs.shape}"
                if kl_inputs["topk_indices"] is not None:
                    assert kl_inputs["topk_indices"].shape == kl_inputs["logits_k"].shape, \
                        f"topk_indices shape {kl_inputs['topk_indices'].shape} != logits_k shape {kl_inputs['logits_k'].shape}"

            return entropy, log_probs, kl_inputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def compute_log_prob_with_logits(self, data: DataProto, kl_topk_k: int) -> tuple:
        """Compute log probability and extract logits for KL divergence computation.

        Args:
            data (DataProto): Input data containing input_ids, attention_mask, etc.
            kl_topk_k: -1 for full vocab logits, >0 for top-k logits.

        Returns:
            log_probs: tensor of shape [batch_size, response_length]
            entropys: tensor of shape [batch_size, response_length]
            kl_inputs: dict containing:
                - logits_k: tensor of shape [batch_size, response_length, vocab_size] if kl_topk_k=-1,
                            else [batch_size, response_length, k]
                - topk_indices: tensor of shape [batch_size, response_length, k] or None if kl_topk_k=-1
                - logsumexp: tensor of shape [batch_size, response_length]
        """
        assert kl_topk_k == -1 or kl_topk_k > 0, f"kl_topk_k must be -1 (full logits) or >0 (top-k), got {kl_topk_k}"
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        kl_inputs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs, kl_inputs = self._forward_micro_batch_with_logits(
                    micro_batch, temperature=temperature, kl_topk_k=kl_topk_k
                )
            log_probs_lst.append(log_probs)
            entropy_lst.append(entropy)
            kl_inputs_lst.append(kl_inputs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0)
        output_kl_inputs = {
            "logits_k": torch.concat([ki["logits_k"] for ki in kl_inputs_lst], dim=0),
            "logsumexp": torch.concat([ki["logsumexp"] for ki in kl_inputs_lst], dim=0),
        }
        if kl_topk_k > 0:
            output_kl_inputs["topk_indices"] = torch.concat([ki["topk_indices"] for ki in kl_inputs_lst], dim=0)
        else:
            output_kl_inputs["topk_indices"] = None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]
            output_kl_inputs["logits_k"] = output_kl_inputs["logits_k"][revert_indices]
            output_kl_inputs["logsumexp"] = output_kl_inputs["logsumexp"][revert_indices]
            if kl_topk_k > 0:
                output_kl_inputs["topk_indices"] = output_kl_inputs["topk_indices"][revert_indices]

        return log_probs, entropys, output_kl_inputs

    def compute_log_prob_at_indices(self, data: DataProto, topk_indices: torch.Tensor) -> tuple:
        """Compute log probability and gather logits at provided top-k indices.

        This is used for:
        - Forward KL: actor update gathers at ref's top-k indices
        - Reverse KL: ref forward gathers at actor's top-k indices

        Args:
            data (DataProto): Input data containing input_ids, attention_mask, etc.
            topk_indices: tensor of shape [batch_size, response_length, k] - indices to gather at.

        Returns:
            log_probs: tensor of shape [batch_size, response_length]
            kl_inputs: dict containing:
                - logits_k: tensor of shape [batch_size, response_length, k]
                - topk_indices: tensor of shape [batch_size, response_length, k] (same as input)
                - logsumexp: tensor of shape [batch_size, response_length]
        """
        assert topk_indices.dim() == 3, f"topk_indices must be 3D [batch, seq, k], got {topk_indices.dim()}D"
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            # Split topk_indices according to indices
            topk_indices_splits = [topk_indices[idx] for idx in indices]
        else:
            micro_batches = batch.split(micro_batch_size)
            topk_indices_splits = list(torch.split(topk_indices, micro_batch_size, dim=0))

        log_probs_lst = []
        kl_inputs_lst = []
        for micro_batch, indices_split in zip(micro_batches, topk_indices_splits):
            with torch.no_grad():
                _, log_probs, kl_inputs = self._forward_micro_batch_with_logits(
                    micro_batch, temperature=temperature, kl_topk_indices=indices_split
                )
            log_probs_lst.append(log_probs)
            kl_inputs_lst.append(kl_inputs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        output_kl_inputs = {
            "logits_k": torch.concat([ki["logits_k"] for ki in kl_inputs_lst], dim=0),
            "topk_indices": torch.concat([ki["topk_indices"] for ki in kl_inputs_lst], dim=0),
            "logsumexp": torch.concat([ki["logsumexp"] for ki in kl_inputs_lst], dim=0),
        }

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            output_kl_inputs["logits_k"] = output_kl_inputs["logits_k"][revert_indices]
            output_kl_inputs["topk_indices"] = output_kl_inputs["topk_indices"][revert_indices]
            output_kl_inputs["logsumexp"] = output_kl_inputs["logsumexp"][revert_indices]

        return log_probs, output_kl_inputs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if 'adv_batch_std' in data.batch:
            select_keys.append('adv_batch_std')
        if self.config.use_kl_loss:
            kl_loss_type = self.config.kl_loss_type
            kl_topk = self.config.get('kl_topk_tokens', None)
            if kl_loss_type in ("full_forward", "full_reverse"):
                # Full KL requires logits
                select_keys.extend(["ref_logits_k", "ref_logsumexp"])
                if kl_topk is not None and kl_topk > 0:
                    if kl_loss_type == "full_reverse":
                        select_keys.append("actor_topk_indices")
                    else:
                        select_keys.append("ref_topk_indices")
                    if self.config.get('kl_use_tail_sampling', True):
                        select_keys.append("ref_log_prob")
            else:
                # Token-level KL
                select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, batch_data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = batch_data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for micro_data in micro_batches:
                micro_data = micro_data.cuda()  # actor device is cpu when using offload
                responses = micro_data['responses']
                response_length = responses.size(1)
                attention_mask = micro_data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = micro_data['loss_mask']
                old_log_prob = micro_data['old_log_probs']
                advantages = micro_data['advantages']

                clip_ratio = self.config.clip_ratio
                clip_ratio_low = self.config.get('clip_ratio_low', None)
                clip_ratio_high = self.config.get('clip_ratio_high', None)
                entropy_coeff = self.config.entropy_coeff

                # Determine if we need logits for full KL
                use_full_kl = (self.config.use_kl_loss and
                               self.config.kl_loss_type in ("full_forward", "full_reverse"))
                kl_topk = self.config.get('kl_topk_tokens', None) if use_full_kl else None

                if use_full_kl and kl_topk is not None and kl_topk > 0:
                    # Memory-efficient top-k mode: need to gather at indices
                    if self.config.kl_loss_type == "full_reverse":
                        kl_topk_indices = micro_data["actor_topk_indices"]
                    else:
                        kl_topk_indices = micro_data["ref_topk_indices"]
                    entropy, log_prob, kl_inputs = self._forward_micro_batch_with_logits(
                        micro_batch=micro_data, temperature=temperature, kl_topk_indices=kl_topk_indices
                    )
                    actor_logits_k = kl_inputs["logits_k"]
                    actor_logsumexp = kl_inputs["logsumexp"]
                elif use_full_kl:
                    # Full logits mode
                    entropy, log_prob, kl_inputs = self._forward_micro_batch_with_logits(
                        micro_batch=micro_data, temperature=temperature, kl_topk_k=-1
                    )
                    actor_logits = kl_inputs["logits_k"]
                    actor_logsumexp = kl_inputs["logsumexp"]
                else:
                    # Standard mode: no logits needed
                    entropy, log_prob = self._forward_micro_batch(micro_batch=micro_data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio,
                                                                              cliprange_low=clip_ratio_low,
                                                                              cliprange_high=clip_ratio_high)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    kl_loss_type = self.config.kl_loss_type
                    kl_topk = self.config.get('kl_topk_tokens', None)
                    use_kl_iw = self.config.get('use_kl_iw', False)

                    # Compute importance weight for off-policy correction
                    if use_kl_iw:
                        log_kl_iw = (log_prob - old_log_prob).detach()
                        log_kl_iw = torch.clamp(log_kl_iw, min=-20, max=20)
                        kl_iw = torch.exp(log_kl_iw)
                        # Apply optional clipping bounds
                        kl_iw_clip_lower = self.config.get('kl_iw_clip_lower', None)
                        kl_iw_clip_upper = self.config.get('kl_iw_clip_upper', None)
                        if kl_iw_clip_lower is not None or kl_iw_clip_upper is not None:
                            kl_iw = torch.clamp(kl_iw, min=kl_iw_clip_lower, max=kl_iw_clip_upper)

                    if kl_loss_type == "full_forward":
                        # Full forward KL: KL(π_ref || π)
                        if kl_topk is not None and kl_topk > 0:
                            use_tail_sampling = self.config.get('kl_use_tail_sampling', True)
                            tail_sampling_kwargs = {}
                            if use_tail_sampling:
                                tail_sampling_kwargs = dict(
                                    ref_topk_indices=micro_data["ref_topk_indices"],
                                    sampled_indices=responses,
                                    log_prob=log_prob,
                                    ref_log_prob=micro_data["ref_log_prob"],
                                )
                            # Returns (L1, L2): L1 is exact head term, L2 is sampled tail term
                            kl_L1, kl_L2 = core_algos.compute_memory_efficient_kl(
                                actor_logits_k=actor_logits_k,
                                actor_logsumexp=actor_logsumexp,
                                ref_logits_k=micro_data["ref_logits_k"],
                                ref_logsumexp=micro_data["ref_logsumexp"],
                                kl_type="full_forward",
                                use_tail_sampling=use_tail_sampling,
                                **tail_sampling_kwargs,
                            )
                            # Apply importance weighting only to L2 (sampled tail term)
                            if use_kl_iw:
                                kl_L2 = kl_L2 * kl_iw
                            kld = kl_L1 + kl_L2
                        else:
                            # Full vocab KL - exact computation, no importance weighting needed
                            kld = core_algos.compute_full_forward_kl(
                                logits=actor_logits, ref_logits=micro_data["ref_logits_k"]
                            )
                    elif kl_loss_type == "full_reverse":
                        # Full reverse KL: KL(π || π_ref)
                        if kl_topk is not None and kl_topk > 0:
                            use_tail_sampling = self.config.get('kl_use_tail_sampling', True)
                            tail_sampling_kwargs = {}
                            if use_tail_sampling:
                                tail_sampling_kwargs = dict(
                                    actor_topk_indices=micro_data["actor_topk_indices"],
                                    sampled_indices=responses,
                                    log_prob=log_prob,
                                    ref_log_prob=micro_data["ref_log_prob"],
                                )
                            # Returns (L1, L2): L1 is exact head term, L2 is sampled tail term
                            kl_L1, kl_L2 = core_algos.compute_memory_efficient_kl(
                                actor_logits_k=actor_logits_k,
                                actor_logsumexp=actor_logsumexp,
                                ref_logits_k=micro_data["ref_logits_k"],
                                ref_logsumexp=micro_data["ref_logsumexp"],
                                kl_type="full_reverse",
                                use_tail_sampling=use_tail_sampling,
                                **tail_sampling_kwargs,
                            )
                            # Apply importance weighting only to L2 (sampled tail term)
                            if use_kl_iw:
                                kl_L2 = kl_L2 * kl_iw
                            kld = kl_L1 + kl_L2
                        else:
                            # Full vocab KL - exact computation, no importance weighting needed
                            kld = core_algos.compute_full_reverse_kl(
                                logits=actor_logits, ref_logits=micro_data["ref_logits_k"]
                            )
                    else:
                        # Token-level KL approximations (k1, k2, k3, k4, k5, k6, low_var_kl, etc.)
                        # These are all sampled estimators, so kl_iw applies to the whole thing
                        ref_log_prob = micro_data['ref_log_prob']
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=kl_loss_type)
                        if use_kl_iw:
                            kld = kld * kl_iw

                    # Normalize KL by advantage batch std if available (before reduction)
                    if ('adv_batch_std' in micro_data) and (micro_data['adv_batch_std'] is not None):
                        adv_batch_std = micro_data['adv_batch_std']
                        assert kld.shape[0] == adv_batch_std.shape[0], \
                            f"Batch size mismatch: kld {kld.shape[0]} vs adv_batch_std {adv_batch_std.shape[0]}"
                        kld = kld / (adv_batch_std.unsqueeze(-1) + 1e-6)
                        metrics['actor/adv_batch_std'] = adv_batch_std.mean().item()
                    kl_loss = masked_mean(kld, response_mask)
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                micro_metrics = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, micro_metrics)

            grad_norm = self._optimizer_step()
            grad_metrics = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, grad_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
