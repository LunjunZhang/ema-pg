# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   norm_adv_by_std_in_grpo: bool = True,
                                   norm_adv_by_batch_std: bool = False):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: `(bool)`
            Whether to normalize by per-prompt std (original GRPO behavior)
        norm_adv_by_batch_std: `(bool)`
            Whether to normalize by batch-wise std. Must be False if norm_adv_by_std_in_grpo is True.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert not (norm_adv_by_std_in_grpo and norm_adv_by_batch_std), \
        "norm_adv_by_batch_std can only be True when norm_adv_by_std_in_grpo is False"
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        # Normalize by batch std after mean subtraction
        # scores is shape (bs,) - one value per sequence
        batch_std = None
        if norm_adv_by_batch_std:
            _batch_std = torch.std(scores)
            if not torch.isnan(_batch_std).any():
                scores = scores / (_batch_std + epsilon)
                batch_std = _batch_std  # only set if not NaN

        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores, batch_std


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange,
                        cliprange_low=None, cliprange_high=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float, optional)
            The lower clip range. If None, defaults to cliprange.
            Controls clipping when ratio < 1 (policy moved away from old policy).
        cliprange_high: (float, optional)
            The upper clip range. If None, defaults to cliprange.
            Controls clipping when ratio > 1 (policy moved toward old policy).

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # Use symmetric clipping if asymmetric bounds not specified
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ('low_var_kl', 'k3'):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    # Sampled forward KL estimator with importance weighting
    # KL(π_ref || π) ≈ E_{a~π}[(π_ref(a)/π(a)) * log(π_ref(a)/π(a))]
    # logw = log(π_ref/π), importance weight = exp(logw) = π_ref/π
    # logr provides gradient flow through π
    if kl_penalty == "k5":
        logw = ref_logprob - logprob  # log(π_ref/π)
        logw = torch.clamp(logw, min=-20, max=20)  # numerical stability
        logr = logprob - logprob.detach()  # = 0 but with gradient flow
        sampled_forward_kl = logw.detach().exp() * logw + logr
        return sampled_forward_kl

    # Sampled reverse KL estimator
    # KL(π || π_ref) ≈ E_{a~π}[log(π(a)/π_ref(a))]
    # pg_ratio provides gradient flow through π
    if kl_penalty == "k4":
        pg_ratio = (logprob - logprob.detach()).exp()  # = 1.0 but with gradient flow
        sampled_reverse_kl = (logprob - ref_logprob).detach() * pg_ratio
        return sampled_reverse_kl

    # K3++: K3 (low_var_kl) times pg_ratio for gradient flow
    # This is the low-variance KL estimator with explicit gradient flow through π
    if kl_penalty == "k3plusplus":
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        kld = torch.clamp(kld, min=-10, max=10)
        pg_ratio = (logprob - logprob.detach()).exp()  # = 1.0 but with gradient flow
        return kld * pg_ratio

    if kl_penalty in ("full_forward", "full_reverse"):
        # Full KL requires logits, not just log probabilities of sampled tokens.
        # Users should set kl_loss_type to "full_forward" or "full_reverse" in config.
        raise ValueError(
            f"kl_penalty='{kl_penalty}' requires full logits, not just log probabilities. "
            "Set actor.kl_loss_type to 'full_forward' (KL(π_ref||π)) or 'full_reverse' (KL(π||π_ref)) "
            "in your config."
        )

    raise NotImplementedError(f"Unknown kl_penalty type: {kl_penalty}")


# =============================================================================
# FULL KL DIVERGENCE COMPUTATION
# =============================================================================

def compute_full_forward_kl(logits: torch.FloatTensor, ref_logits: torch.FloatTensor) -> torch.FloatTensor:
    """Compute full forward KL divergence: KL(π_ref || π).

    Forward KL(π_ref || π) = Σ_v π_ref(v) * (log π_ref(v) - log π(v))
                           = -H(π_ref) + CE(π_ref, π)

    This is the "mode-covering" KL that penalizes the current policy
    for having low probability where the reference has high probability.

    Args:
        logits: Logits from current policy, shape (batch_size, seq_length, vocab_size)
        ref_logits: Logits from reference policy, shape (batch_size, seq_length, vocab_size)

    Returns:
        kl: Forward KL divergence per token, shape (batch_size, seq_length)
    """
    # Compute probabilities from reference policy
    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    ref_probs = torch.exp(ref_log_probs)

    # Compute log probabilities from current policy
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Forward KL = Σ_v p_ref(v) * (log p_ref(v) - log p(v))
    kl = torch.sum(ref_probs * (ref_log_probs - log_probs), dim=-1)

    return kl


def compute_full_reverse_kl(logits: torch.FloatTensor, ref_logits: torch.FloatTensor) -> torch.FloatTensor:
    """Compute full reverse KL divergence: KL(π || π_ref).

    Reverse KL(π || π_ref) = Σ_v π(v) * (log π(v) - log π_ref(v))
                           = -H(π) + CE(π, π_ref)

    This is the "mode-seeking" KL commonly used in RL that penalizes
    the current policy for having high probability where the reference
    has low probability.

    Args:
        logits: Logits from current policy, shape (batch_size, seq_length, vocab_size)
        ref_logits: Logits from reference policy, shape (batch_size, seq_length, vocab_size)

    Returns:
        kl: Reverse KL divergence per token, shape (batch_size, seq_length)
    """
    # Compute probabilities from current policy
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Compute log probabilities from reference policy
    ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

    # Reverse KL = Σ_v p(v) * (log p(v) - log q(v))
    kl = torch.sum(probs * (log_probs - ref_log_probs), dim=-1)

    return kl


# =============================================================================
# MEMORY-EFFICIENT TOP-K KL DIVERGENCE COMPUTATION
# =============================================================================
#
# This section implements memory-efficient KL divergence computation for RLHF.
# Instead of storing full vocab logits (~9.1 GB per micro_batch for 150k vocab),
# we store only top-k logits + logsumexp (~30 MB for k=1024), reducing memory by ~99%.
#
# FOR REVERSE KL (KL(π || π_ref)) - expectation under actor:
#   Step 1: Actor computes top-k indices (where actor probability mass is concentrated)
#   Step 2: Ref gathers logits at ACTOR's top-k indices
#   Step 3: Actor update uses pre-gathered logits for KL computation
#
# FOR FORWARD KL (KL(π_ref || π)) - expectation under ref:
#   Step 1: Ref computes top-k indices (where ref probability mass is concentrated)
#   Step 2: Actor gathers logits at REF's top-k indices during update
#
# =============================================================================


def compute_not_in_topk_mask(
    sampled_indices: torch.LongTensor,
    topk_indices: torch.LongTensor,
) -> torch.BoolTensor:
    """Return mask that is True where sampled token is NOT in top-k.

    Args:
        sampled_indices: The sampled token indices (responses), shape (batch, seq).
        topk_indices: The top-k indices from actor, shape (batch, seq, k).

    Returns:
        Mask tensor of shape (batch, seq), True where sampled token is NOT in top-k.
    """
    # Shape validation
    assert sampled_indices.dim() == 2, f"sampled_indices must be 2D, got {sampled_indices.dim()}D"
    assert topk_indices.dim() == 3, f"topk_indices must be 3D, got {topk_indices.dim()}D"
    batch, seq = sampled_indices.shape
    assert topk_indices.shape[:2] == (batch, seq), \
        f"topk_indices shape {topk_indices.shape[:2]} != sampled_indices shape {(batch, seq)}"

    # Check if sampled token matches any of the k indices
    # sampled_indices: (batch, seq) -> (batch, seq, 1) for broadcasting
    # topk_indices: (batch, seq, k)
    in_topk = (topk_indices == sampled_indices.unsqueeze(-1)).any(dim=-1)  # (batch, seq)
    return ~in_topk  # True if NOT in top-k


def compute_memory_efficient_kl(
    actor_logits_k: torch.FloatTensor,
    actor_logsumexp: torch.FloatTensor,
    ref_logits_k: torch.FloatTensor,
    ref_logsumexp: torch.FloatTensor,
    kl_type: str = "full_reverse",
    # Optional parameters for tail sampling (stratified KL):
    use_tail_sampling: bool = False,  # Default False for backward compatibility
    actor_topk_indices: torch.LongTensor = None,  # (batch, seq, k) - for full_reverse
    ref_topk_indices: torch.LongTensor = None,  # (batch, seq, k) - for full_forward
    sampled_indices: torch.LongTensor = None,  # (batch, seq) - response tokens
    log_prob: torch.FloatTensor = None,  # (batch, seq) - current actor log prob at sampled token
    ref_log_prob: torch.FloatTensor = None,  # (batch, seq) - ref log prob at sampled token
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """Memory-efficient KL computation using pre-gathered logits.

    Supports optional tail sampling correction that combines:
    - Head term (L1): Exact KL over top-K tokens
    - Tail term (L2): Sampled KL correction for tokens outside top-K

    For full_reverse KL(π || π_ref): mask uses actor_topk_indices, L2 uses reverse sampled KL
    For full_forward KL(π_ref || π): mask uses ref_topk_indices, L2 uses K4 sampled forward KL

    Returns L1 and L2 separately so that importance weighting (kl_iw) can be applied
    only to L2 (the sampled tail term), not to L1 (the exact head term).

    Args:
        actor_logits_k: Pre-gathered actor logits at top-k indices, shape (batch, seq, k).
        actor_logsumexp: Logsumexp of full actor logits, shape (batch, seq).
        ref_logits_k: Pre-gathered ref logits at top-k indices, shape (batch, seq, k).
        ref_logsumexp: Logsumexp of full ref logits, shape (batch, seq).
        kl_type: "full_forward" for KL(π_ref || π) or "full_reverse" for KL(π || π_ref).
        use_tail_sampling: Whether to use tail sampling correction.
        actor_topk_indices: Top-k indices from actor, shape (batch, seq, k).
            Required for full_reverse with use_tail_sampling=True.
        ref_topk_indices: Top-k indices from ref, shape (batch, seq, k).
            Required for full_forward with use_tail_sampling=True.
        sampled_indices: Sampled token indices (responses), shape (batch, seq).
            Required if use_tail_sampling=True.
        log_prob: Current actor log prob at sampled token, shape (batch, seq).
            Required if use_tail_sampling=True.
        ref_log_prob: Ref log prob at sampled token, shape (batch, seq).
            Required if use_tail_sampling=True.

    Returns:
        L1: Head term (exact KL over top-K), shape (batch, seq).
        L2: Tail term (sampled KL correction), shape (batch, seq). Zero if use_tail_sampling=False.
    """
    # Shape validation: all tensors must have matching batch and seq dimensions
    assert actor_logits_k.dim() == 3, f"actor_logits_k must be 3D, got {actor_logits_k.dim()}D"
    assert ref_logits_k.dim() == 3, f"ref_logits_k must be 3D, got {ref_logits_k.dim()}D"
    assert actor_logsumexp.dim() == 2, f"actor_logsumexp must be 2D, got {actor_logsumexp.dim()}D"
    assert ref_logsumexp.dim() == 2, f"ref_logsumexp must be 2D, got {ref_logsumexp.dim()}D"

    batch, seq, k = actor_logits_k.shape
    assert ref_logits_k.shape == (batch, seq, k), \
        f"ref_logits_k shape {ref_logits_k.shape} != actor_logits_k shape {actor_logits_k.shape}"
    assert actor_logsumexp.shape == (batch, seq), \
        f"actor_logsumexp shape {actor_logsumexp.shape} != expected ({batch}, {seq})"
    assert ref_logsumexp.shape == (batch, seq), \
        f"ref_logsumexp shape {ref_logsumexp.shape} != expected ({batch}, {seq})"

    # Common tail sampling validations (shared by full_forward and full_reverse)
    if use_tail_sampling:
        assert sampled_indices is not None, "sampled_indices required when use_tail_sampling=True"
        assert log_prob is not None, "log_prob required when use_tail_sampling=True"
        assert ref_log_prob is not None, "ref_log_prob required when use_tail_sampling=True"

        # Shape validation for common tail sampling tensors
        assert sampled_indices.shape == (batch, seq), \
            f"sampled_indices shape {sampled_indices.shape} != expected ({batch}, {seq})"
        assert log_prob.shape == (batch, seq), \
            f"log_prob shape {log_prob.shape} != expected ({batch}, {seq})"
        assert ref_log_prob.shape == (batch, seq), \
            f"ref_log_prob shape {ref_log_prob.shape} != expected ({batch}, {seq})"

    actor_log_probs_k = actor_logits_k - actor_logsumexp.unsqueeze(-1)
    ref_log_probs_k = ref_logits_k - ref_logsumexp.unsqueeze(-1)

    if kl_type == "full_forward":
        # full_forward: KL(π_ref || π)
        ref_probs_k = ref_log_probs_k.exp()
        L1 = (ref_probs_k * (ref_log_probs_k - actor_log_probs_k)).sum(dim=-1)

        # Tail sampling correction using K4 sampled forward KL
        if use_tail_sampling:
            assert ref_topk_indices is not None, \
                "ref_topk_indices required for full_forward with use_tail_sampling=True"
            assert ref_topk_indices.shape == (batch, seq, k), \
                f"ref_topk_indices shape {ref_topk_indices.shape} != expected ({batch}, {seq}, {k})"

            # Compute mask: True if sampled token NOT in ref's top-k
            not_in_topk = compute_not_in_topk_mask(sampled_indices, ref_topk_indices)

            # K4 sampled forward KL: logw.detach().exp() * logw + logr
            logw = ref_log_prob - log_prob  # log(π_ref/π)
            logw = torch.clamp(logw, min=-20, max=20)  # numerical stability
            logr = log_prob - log_prob.detach()  # = 0 but with gradient flow
            sampled_forward_kl = logw.detach().exp() * logw + logr

            L2 = not_in_topk.to(dtype=sampled_forward_kl.dtype) * sampled_forward_kl
        else:
            L2 = torch.zeros_like(L1)
    else:
        # full_reverse: KL(π || π_ref)
        actor_probs_k = actor_log_probs_k.exp()
        L1 = (actor_probs_k * (actor_log_probs_k - ref_log_probs_k)).sum(dim=-1)

        # Tail sampling correction using reverse sampled KL
        if use_tail_sampling:
            assert actor_topk_indices is not None, \
                "actor_topk_indices required for full_reverse with use_tail_sampling=True"
            assert actor_topk_indices.shape == (batch, seq, k), \
                f"actor_topk_indices shape {actor_topk_indices.shape} != expected ({batch}, {seq}, {k})"

            # Compute mask: True if sampled token NOT in actor's top-k
            not_in_topk = compute_not_in_topk_mask(sampled_indices, actor_topk_indices)

            # L2 = mask * pg_ratio * sg(log_prob - ref_log_prob)
            pg_ratio = (log_prob - log_prob.detach()).exp()  # = 1.0 but with gradient flow
            kl_at_sampled = (log_prob - ref_log_prob).detach() * pg_ratio

            L2 = not_in_topk.to(dtype=kl_at_sampled.dtype) * kl_at_sampled
        else:
            L2 = torch.zeros_like(L1)

    return L1, L2
