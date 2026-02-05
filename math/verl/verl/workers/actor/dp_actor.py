# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_full_forward_kl,
    compute_full_reverse_kl,
    compute_memory_efficient_kl,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _forward_micro_batch_with_logits(
        self, micro_batch, temperature, calculate_entropy=False,
        kl_topk_k: int | None = None, kl_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass that also returns logits for KL divergence computation.

        This is a separate function from _forward_micro_batch to maintain backward compatibility.
        Use this function when you need logits for full/top-k KL computation.

        Args:
            micro_batch: Input batch dict.
            temperature: Temperature for logits.
            calculate_entropy: Whether to compute entropy.
            kl_topk_k: -1 for full logits, >0 to compute top-k from own logits.
                       None means use kl_topk_indices instead.
            kl_topk_indices: Indices to gather at (only used when kl_topk_k is None).

        Returns:
            entropy: (bs, response_len) or None if calculate_entropy=False
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

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo
                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                if self.use_fused_kernels:
                    raise ValueError(
                        "use_fused_kernels=True is not compatible with _forward_micro_batch_with_logits. "
                        "Fused kernels compute log_probs and entropy directly without materializing logits. "
                        "To use full KL divergence, set use_fused_kernels=False."
                    )

                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)

                # Use inplace_backward when possible (entropy checkpointed or not computed)
                # This avoids saving full logits for log_probs backward
                use_inplace_backward = not calculate_entropy or self.config.entropy_checkpointing
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=use_inplace_backward,
                )

                if calculate_entropy:
                    if not self.config.entropy_checkpointing:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

                # Derive logsumexp from log_probs: logsumexp = logits[label] - log_softmax(logits)[label]
                # This avoids calling logsumexp() which would save full logits for backward
                logits_at_label_rmpad = logits_rmpad.gather(-1, input_ids_rmpad_rolled.unsqueeze(-1)).squeeze(-1)
                logsumexp_rmpad = logits_at_label_rmpad - log_probs
                if kl_topk_k == -1:
                    logits_k_rmpad = logits_rmpad
                    topk_indices_rmpad = None
                elif kl_topk_k is not None and kl_topk_k > 0:
                    _, topk_indices_rmpad = logits_rmpad.topk(kl_topk_k, dim=-1)
                    logits_k_rmpad = logits_rmpad.gather(-1, topk_indices_rmpad)
                else:
                    # kl_topk_k is None, use kl_topk_indices
                    # Don't keep full logits - we'll gather directly at response positions later
                    logits_k_rmpad = None
                    topk_indices_rmpad = None

                # gather if sp > 1
                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if logits_k_rmpad is not None:
                        logits_k_rmpad = gather_outputs_and_unpad(logits_k_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    else:
                        # For kl_topk_indices case, gather full logits for later direct indexing
                        logits_rmpad = gather_outputs_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    logsumexp_rmpad = gather_outputs_and_unpad(logsumexp_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if topk_indices_rmpad is not None:
                        topk_indices_rmpad = gather_outputs_and_unpad(topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # Handle kl_topk_indices case: gather directly at response positions
                # Memory-efficient: only gathers k logits per position, never materializes full vocab
                if kl_topk_k is None and kl_topk_indices is not None:
                    # Validate kl_topk_indices shape
                    assert kl_topk_indices.shape[0] == batch_size, \
                        f"kl_topk_indices batch size {kl_topk_indices.shape[0]} != expected {batch_size}"
                    assert kl_topk_indices.shape[1] == response_length, \
                        f"kl_topk_indices response_length {kl_topk_indices.shape[1]} != expected {response_length}"

                    # Create inverse mapping: padded position -> rmpad position
                    total_nnz = logsumexp_rmpad.shape[0]
                    inverse_indices = torch.full((batch_size * seqlen,), -1, dtype=torch.long, device=logsumexp_rmpad.device)
                    inverse_indices[indices] = torch.arange(total_nnz, device=logsumexp_rmpad.device)

                    # Compute rmpad positions for response tokens
                    # Response is at positions [seqlen - response_length - 1, seqlen - 1) in each sequence
                    response_start = seqlen - response_length - 1
                    batch_offsets = torch.arange(batch_size, device=logsumexp_rmpad.device) * seqlen
                    response_offsets = torch.arange(response_length, device=logsumexp_rmpad.device)
                    flattened_response_pos = batch_offsets.unsqueeze(1) + response_start + response_offsets.unsqueeze(0)
                    rmpad_response_pos = inverse_indices[flattened_response_pos]  # (bsz, response_length)

                    # Handle padding positions: create a mask for valid (non-padding) positions
                    valid_mask = rmpad_response_pos >= 0  # (bsz, response_length)

                    # Replace -1 with 0 for safe indexing (will be masked out later)
                    safe_rmpad_pos = rmpad_response_pos.clamp(min=0)

                    # Gather logits_k using advanced indexing
                    k = kl_topk_indices.shape[-1]
                    rmpad_pos_expanded = safe_rmpad_pos.unsqueeze(-1).expand(-1, -1, k)  # (bsz, response_length, k)
                    logits_k = logits_rmpad[rmpad_pos_expanded.reshape(-1), kl_topk_indices.reshape(-1)]
                    logits_k = logits_k.reshape(batch_size, response_length, k)

                    # Gather logsumexp directly at response positions
                    logsumexp = logsumexp_rmpad[safe_rmpad_pos]  # (bsz, response_length)

                    # Zero out padding positions (they'll be masked in loss computation anyway)
                    logits_k = logits_k * valid_mask.unsqueeze(-1)
                    logsumexp = logsumexp * valid_mask

                    kl_inputs = {"logits_k": logits_k, "topk_indices": kl_topk_indices, "logsumexp": logsumexp}

                    # Pad log_probs and entropy (these are small: just scalar per position)
                    if calculate_entropy:
                        full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                    full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]

                else:
                    # Standard path: pad back to (bsz, seqlen), then slice
                    if calculate_entropy:
                        full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logits_k = pad_input(hidden_states=logits_k_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
                    full_logsumexp = pad_input(hidden_states=logsumexp_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    if topk_indices_rmpad is not None:
                        full_topk_indices = pad_input(hidden_states=topk_indices_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)

                    # only return response part
                    if calculate_entropy:
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                    logits_k = full_logits_k[:, -response_length - 1 : -1, :]
                    logsumexp = full_logsumexp.squeeze(-1)[:, -response_length - 1 : -1]

                    if kl_topk_k == -1:
                        kl_inputs = {"logits_k": logits_k, "topk_indices": None, "logsumexp": logsumexp}
                    else:  # kl_topk_k > 0
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]
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
                    **multi_modal_inputs,
                    use_cache=False,
                )

                if self.use_fused_kernels:
                    raise ValueError(
                        "use_fused_kernels=True is not compatible with _forward_micro_batch_with_logits. "
                        "Fused kernels compute log_probs and entropy directly without materializing logits. "
                        "To use full KL divergence, set use_fused_kernels=False."
                    )

                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]

                # Use inplace_backward when possible (entropy checkpointed or not computed)
                use_inplace_backward = not calculate_entropy or self.config.entropy_checkpointing
                log_probs = logprobs_from_logits(logits, micro_batch["responses"], inplace_backward=use_inplace_backward)

                if calculate_entropy:
                    if not self.config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                # Derive logsumexp from log_probs: logsumexp = logits[label] - log_softmax(logits)[label]
                # This avoids calling logsumexp() which would save full logits for backward
                logits_at_label = logits.gather(-1, micro_batch["responses"].unsqueeze(-1)).squeeze(-1)
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
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

            calculate_entropy: Whether to compute entropy.

        Returns:
            log_probs: tensor of shape [batch_size, response_length]
            entropys: tensor of shape [batch_size, response_length] or None
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_with_logits(self, data: DataProto, kl_topk_k: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute log probability and extract logits for KL divergence computation.

        This unified function handles both:
        - Full logits (kl_topk_k=-1): returns full vocab logits, topk_indices=None
        - Top-k logits (kl_topk_k>0): returns top-k logits and indices

        Args:
            data (DataProto): a DataProto containing input data.
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

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        kl_inputs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, kl_inputs = self._forward_micro_batch_with_logits(
                    model_inputs, temperature=temperature, calculate_entropy=True,
                    kl_topk_k=kl_topk_k
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
        # topk_indices is None when kl_topk_k=-1 (full logits mode)
        if kl_topk_k > 0:
            output_kl_inputs["topk_indices"] = torch.concat([ki["topk_indices"] for ki in kl_inputs_lst], dim=0)
        else:
            output_kl_inputs["topk_indices"] = None

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            entropys = restore_dynamic_batch(entropys, batch_idx_list)
            output_kl_inputs["logits_k"] = restore_dynamic_batch(output_kl_inputs["logits_k"], batch_idx_list)
            output_kl_inputs["logsumexp"] = restore_dynamic_batch(output_kl_inputs["logsumexp"], batch_idx_list)
            if kl_topk_k > 0:
                output_kl_inputs["topk_indices"] = restore_dynamic_batch(output_kl_inputs["topk_indices"], batch_idx_list)

        # Shape validation for output consistency
        batch_size, seq_len = log_probs.shape
        assert entropys.shape == (batch_size, seq_len), \
            f"entropys shape {entropys.shape} != log_probs shape {log_probs.shape}"
        assert output_kl_inputs["logsumexp"].shape == (batch_size, seq_len), \
            f"logsumexp shape {output_kl_inputs['logsumexp'].shape} != ({batch_size}, {seq_len})"
        assert output_kl_inputs["logits_k"].shape[0] == batch_size and output_kl_inputs["logits_k"].shape[1] == seq_len, \
            f"logits_k shape {output_kl_inputs['logits_k'].shape} batch/seq mismatch with ({batch_size}, {seq_len})"
        if kl_topk_k > 0:
            assert output_kl_inputs["topk_indices"].shape == output_kl_inputs["logits_k"].shape, \
                f"topk_indices shape {output_kl_inputs['topk_indices'].shape} != logits_k shape {output_kl_inputs['logits_k'].shape}"
            assert output_kl_inputs["logits_k"].shape[2] == kl_topk_k, \
                f"logits_k last dim {output_kl_inputs['logits_k'].shape[2]} != kl_topk_k {kl_topk_k}"

        return log_probs, entropys, output_kl_inputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob_at_indices(self, data: DataProto, topk_indices: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute log probability and gather logits at provided top-k indices.

        This is used for:
        - Forward KL: actor update gathers at ref's top-k indices
        - Reverse KL: ref forward gathers at actor's top-k indices

        Args:
            data (DataProto): a DataProto containing input data.
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

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split topk_indices along with data
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            # Split topk_indices according to batch_idx_list
            topk_indices_splits = [topk_indices[idx] for idx in batch_idx_list]
        else:
            micro_batches = data.split(micro_batch_size)
            # Split topk_indices to match micro_batches
            topk_indices_splits = list(torch.split(topk_indices, micro_batch_size, dim=0))

        log_probs_lst = []
        kl_inputs_lst = []
        for micro_batch, indices_split in zip(micro_batches, topk_indices_splits):
            micro_batch = micro_batch.to(get_device_id())
            indices_split = indices_split.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, log_probs, kl_inputs = self._forward_micro_batch_with_logits(
                    model_inputs, temperature=temperature, calculate_entropy=False,
                    kl_topk_indices=indices_split
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
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            output_kl_inputs["logits_k"] = restore_dynamic_batch(output_kl_inputs["logits_k"], batch_idx_list)
            output_kl_inputs["topk_indices"] = restore_dynamic_batch(output_kl_inputs["topk_indices"], batch_idx_list)
            output_kl_inputs["logsumexp"] = restore_dynamic_batch(output_kl_inputs["logsumexp"], batch_idx_list)

        return log_probs, output_kl_inputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            if self.config.kl_loss_type in ("full_forward", "full_reverse"):
                kl_topk = self.config.kl_topk_tokens
                # Both modes (full logits and top-k) use ref_logits_k and ref_logsumexp
                # ref_logits_k contains full vocab logits when kl_topk_k=-1, or top-k logits when kl_topk_k>0
                select_keys.extend(["ref_logits_k", "ref_logsumexp"])
                if kl_topk is not None and kl_topk > 0:
                    # Memory-efficient top-k mode: also need indices
                    if self.config.kl_loss_type == "full_reverse":
                        # Reverse KL: indices come from actor
                        select_keys.append("actor_topk_indices")
                    else:
                        # Forward KL: indices come from ref
                        select_keys.append("ref_topk_indices")
                    # For tail sampling, also need ref_log_prob at sampled tokens
                    # (responses and old_log_probs are already in base select_keys)
                    if self.config.kl_use_tail_sampling:
                        select_keys.append("ref_log_prob")
            else:
                select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # Include adv_batch_std for normalizing KL loss by batch std
        if "adv_batch_std" in data.batch.keys():
            select_keys.append("adv_batch_std")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    # Determine which forward function to use based on KL configuration
                    # Three cases:
                    # 1. No full KL (approximate KL or no KL) → _forward_micro_batch (original, no logits)
                    # 2. Full KL with top-k → _forward_micro_batch_with_logits with kl_topk_indices
                    # 3. Full KL without top-k → _forward_micro_batch_with_logits with kl_topk_k=-1
                    need_full_kl = self.config.use_kl_loss and self.config.kl_loss_type in ("full_forward", "full_reverse")
                    kl_topk = getattr(self.config, "kl_topk_tokens", None)
                    use_topk_kl = need_full_kl and kl_topk is not None and kl_topk > 0

                    if use_topk_kl:
                        # Case 2: Full KL with top-k - gather at stored indices
                        if self.config.kl_loss_type == "full_reverse":
                            # Reverse KL: use actor's top-k indices
                            topk_indices = model_inputs["actor_topk_indices"]
                        else:
                            # Forward KL: use ref's top-k indices
                            topk_indices = model_inputs["ref_topk_indices"]
                        entropy, log_prob, kl_inputs = self._forward_micro_batch_with_logits(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            kl_topk_indices=topk_indices
                        )
                        actor_logits_k = kl_inputs["logits_k"]
                        actor_logsumexp = kl_inputs["logsumexp"]
                    elif need_full_kl:
                        # Case 3: Full KL without top-k - need full vocab logits
                        entropy, log_prob, kl_inputs = self._forward_micro_batch_with_logits(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            kl_topk_k=-1
                        )
                        actor_logits = kl_inputs["logits_k"]
                    else:
                        # Case 1: No full KL - use original function without logits
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        actor_logits = None

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        # =============================================================
                        # KL LOSS COMPUTATION
                        # =============================================================
                        # Three modes: approximate KL, full KL with top-k, full KL with full logits.
                        #
                        # MEMORY-EFFICIENT TOP-K MODE (kl_topk_tokens > 0):
                        # All logits are pre-gathered at top-k indices, reducing memory by ~99%.
                        # The logsumexp values are computed over FULL vocab before gathering,
                        # ensuring correct probability normalization.
                        #
                        # REVERSE KL (KL(π || π_ref)) with top-k:
                        #   - actor_topk_indices: from OLD actor (computed in ray_trainer)
                        #   - actor_logits_k, actor_logsumexp: fresh actor logits gathered at these indices
                        #   - ref_logits_k, ref_logsumexp: ref logits gathered at same indices
                        #
                        # FORWARD KL (KL(π_ref || π)) with top-k:
                        #   - ref_topk_indices: from ref policy (computed in ray_trainer)
                        #   - actor_logits_k, actor_logsumexp: fresh actor logits gathered at ref's indices
                        #   - ref_logits_k, ref_logsumexp: ref's own top-k logits
                        #
                        # FULL LOGITS MODE (kl_topk_tokens <= 0 or None):
                        # Uses full vocab logits. More accurate but higher memory usage.
                        # =============================================================
                        kl_topk = self.config.kl_topk_tokens
                        use_kl_iw = self.config.use_kl_iw

                        # Compute importance weight for off-policy correction
                        if use_kl_iw:
                            old_log_prob = model_inputs["old_log_probs"]
                            log_kl_iw = (log_prob - old_log_prob).detach()
                            log_kl_iw = torch.clamp(log_kl_iw, min=-20, max=20)
                            kl_iw = torch.exp(log_kl_iw)
                            # Apply optional clipping to kl_iw
                            kl_iw_clip_lower = self.config.get('kl_iw_clip_lower', None)
                            kl_iw_clip_upper = self.config.get('kl_iw_clip_upper', None)
                            if kl_iw_clip_lower is not None or kl_iw_clip_upper is not None:
                                kl_iw = torch.clamp(kl_iw, min=kl_iw_clip_lower, max=kl_iw_clip_upper)

                        if self.config.kl_loss_type == "full_forward":
                            # Full forward KL: KL(π_ref || π)
                            if kl_topk is not None and kl_topk > 0:
                                # Validate required keys and shapes for memory-efficient KL
                                assert "ref_logits_k" in model_inputs, "ref_logits_k missing from model_inputs"
                                assert "ref_logsumexp" in model_inputs, "ref_logsumexp missing from model_inputs"
                                assert actor_logits_k.shape == model_inputs["ref_logits_k"].shape, \
                                    f"Shape mismatch: actor_logits_k {actor_logits_k.shape} vs ref_logits_k {model_inputs['ref_logits_k'].shape}"

                                # Prepare tail sampling params if enabled
                                use_tail_sampling = self.config.kl_use_tail_sampling
                                tail_sampling_kwargs = {}
                                if use_tail_sampling:
                                    # Validate shapes for tail sampling
                                    responses = model_inputs["responses"]
                                    assert responses.dim() == 2, \
                                        f"responses must be 2D (batch, response_len), got {responses.dim()}D"
                                    batch_size, response_len = responses.shape
                                    assert actor_logits_k.shape[0] == batch_size, \
                                        f"Batch mismatch: actor_logits_k {actor_logits_k.shape[0]} vs responses {batch_size}"
                                    assert actor_logits_k.shape[1] == response_len, \
                                        f"Seq length mismatch: actor_logits_k seq={actor_logits_k.shape[1]} vs responses response_len={response_len}"

                                    tail_sampling_kwargs = dict(
                                        ref_topk_indices=model_inputs["ref_topk_indices"],
                                        sampled_indices=responses,
                                        log_prob=log_prob,
                                        ref_log_prob=model_inputs["ref_log_prob"],
                                    )

                                # Returns (L1, L2): L1 is exact head term, L2 is sampled tail term
                                kl_L1, kl_L2 = compute_memory_efficient_kl(
                                    actor_logits_k=actor_logits_k,
                                    actor_logsumexp=actor_logsumexp,
                                    ref_logits_k=model_inputs["ref_logits_k"],
                                    ref_logsumexp=model_inputs["ref_logsumexp"],
                                    kl_type="full_forward",
                                    use_tail_sampling=use_tail_sampling,
                                    **tail_sampling_kwargs,
                                )
                                # Apply importance weighting only to L2 (sampled tail term)
                                if use_kl_iw:
                                    kl_L2 = kl_L2 * kl_iw
                                kld = kl_L1 + kl_L2
                            else:
                                # Full logits mode: ref_logits_k contains full vocab logits
                                # Exact computation, no importance weighting needed
                                assert "ref_logits_k" in model_inputs, "ref_logits_k missing from model_inputs"
                                ref_logits = model_inputs["ref_logits_k"]
                                assert actor_logits.shape == ref_logits.shape, \
                                    f"Shape mismatch: actor_logits {actor_logits.shape} vs ref_logits {ref_logits.shape}"
                                kld = compute_full_forward_kl(logits=actor_logits, ref_logits=ref_logits)
                        elif self.config.kl_loss_type == "full_reverse":
                            # Full reverse KL: KL(π || π_ref)
                            if kl_topk is not None and kl_topk > 0:
                                # Validate required keys and shapes for memory-efficient KL
                                assert "ref_logits_k" in model_inputs, "ref_logits_k missing from model_inputs"
                                assert "ref_logsumexp" in model_inputs, "ref_logsumexp missing from model_inputs"
                                assert actor_logits_k.shape == model_inputs["ref_logits_k"].shape, \
                                    f"Shape mismatch: actor_logits_k {actor_logits_k.shape} vs ref_logits_k {model_inputs['ref_logits_k'].shape}"

                                # Prepare tail sampling params if enabled
                                use_tail_sampling = self.config.kl_use_tail_sampling
                                tail_sampling_kwargs = {}
                                if use_tail_sampling:
                                    # Validate shapes for tail sampling
                                    # actor_logits_k: (batch, seq, k), responses: (batch, response_len)
                                    # seq dimension should match response_len
                                    responses = model_inputs["responses"]
                                    assert responses.dim() == 2, \
                                        f"responses must be 2D (batch, response_len), got {responses.dim()}D"
                                    batch_size, response_len = responses.shape
                                    assert actor_logits_k.shape[0] == batch_size, \
                                        f"Batch mismatch: actor_logits_k {actor_logits_k.shape[0]} vs responses {batch_size}"
                                    assert actor_logits_k.shape[1] == response_len, \
                                        f"Seq length mismatch: actor_logits_k seq={actor_logits_k.shape[1]} vs responses response_len={response_len}"

                                    tail_sampling_kwargs = dict(
                                        actor_topk_indices=model_inputs["actor_topk_indices"],
                                        sampled_indices=responses,
                                        log_prob=log_prob,
                                        ref_log_prob=model_inputs["ref_log_prob"],
                                    )

                                # Returns (L1, L2): L1 is exact head term, L2 is sampled tail term
                                kl_L1, kl_L2 = compute_memory_efficient_kl(
                                    actor_logits_k=actor_logits_k,
                                    actor_logsumexp=actor_logsumexp,
                                    ref_logits_k=model_inputs["ref_logits_k"],
                                    ref_logsumexp=model_inputs["ref_logsumexp"],
                                    kl_type="full_reverse",
                                    use_tail_sampling=use_tail_sampling,
                                    **tail_sampling_kwargs,
                                )
                                # Apply importance weighting only to L2 (sampled tail term)
                                if use_kl_iw:
                                    kl_L2 = kl_L2 * kl_iw
                                kld = kl_L1 + kl_L2
                            else:
                                # Full logits mode: ref_logits_k contains full vocab logits
                                # Exact computation, no importance weighting needed
                                assert "ref_logits_k" in model_inputs, "ref_logits_k missing from model_inputs"
                                ref_logits = model_inputs["ref_logits_k"]
                                assert actor_logits.shape == ref_logits.shape, \
                                    f"Shape mismatch: actor_logits {actor_logits.shape} vs ref_logits {ref_logits.shape}"
                                kld = compute_full_reverse_kl(logits=actor_logits, ref_logits=ref_logits)
                        else:
                            # Token-level KL approximations (k1, k2, k3, low_var_kl, etc.)
                            # These are all sampled estimators, so kl_iw applies to the whole thing
                            ref_log_prob = model_inputs["ref_log_prob"]
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            if use_kl_iw:
                                kld = kld * kl_iw

                        # Normalize KL by advantage batch std if available (before reduction)
                        if "adv_batch_std" in model_inputs and model_inputs["adv_batch_std"] is not None:
                            adv_batch_std = model_inputs["adv_batch_std"]
                            assert kld.shape[0] == adv_batch_std.shape[0], \
                                f"Batch size mismatch: kld {kld.shape[0]} vs adv_batch_std {adv_batch_std.shape[0]}"
                            kld = kld / (adv_batch_std.unsqueeze(-1) + 1e-6)
                            micro_batch_metrics["actor/adv_batch_std"] = adv_batch_std.mean().item()

                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
