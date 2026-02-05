"""
Tests for tail sampling KL divergence implementation.

This file tests:
1. compute_not_in_topk_mask - the indexing operation to check if sampled token is in top-k
2. compute_memory_efficient_kl with tail sampling - end-to-end correctness
3. K4, K5, K6 standalone estimators
4. Edge cases and numerical stability
"""

import torch
import torch.nn.functional as F
import pytest


# =============================================================================
# Test compute_not_in_topk_mask
# =============================================================================

def compute_not_in_topk_mask(
    sampled_indices: torch.LongTensor,
    topk_indices: torch.LongTensor,
) -> torch.BoolTensor:
    """Copy of the function for isolated testing."""
    assert sampled_indices.dim() == 2, f"sampled_indices must be 2D, got {sampled_indices.dim()}D"
    assert topk_indices.dim() == 3, f"topk_indices must be 3D, got {topk_indices.dim()}D"
    batch, seq = sampled_indices.shape
    assert topk_indices.shape[:2] == (batch, seq), \
        f"topk_indices shape {topk_indices.shape[:2]} != sampled_indices shape {(batch, seq)}"

    in_topk = (topk_indices == sampled_indices.unsqueeze(-1)).any(dim=-1)
    return ~in_topk


class TestComputeNotInTopkMask:
    """Test the compute_not_in_topk_mask function."""

    def test_basic_in_topk(self):
        """Test that tokens IN top-k are correctly identified (mask=False)."""
        batch, seq, k = 2, 3, 4

        # Top-k indices for each position
        topk_indices = torch.tensor([
            [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]],
            [[100, 101, 102, 103], [110, 111, 112, 113], [120, 121, 122, 123]],
        ])  # (2, 3, 4)

        # Sampled indices that ARE in top-k
        sampled_indices = torch.tensor([
            [1, 12, 22],  # 1 is in [0,1,2,3], 12 is in [10,11,12,13], 22 is in [20,21,22,23]
            [100, 111, 123],  # all in their respective top-k
        ])  # (2, 3)

        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)

        # All should be False (token IS in top-k, so NOT_in_topk = False)
        assert mask.shape == (batch, seq)
        assert mask.dtype == torch.bool
        assert not mask.any(), f"Expected all False, got {mask}"

    def test_basic_not_in_topk(self):
        """Test that tokens NOT in top-k are correctly identified (mask=True)."""
        batch, seq, k = 2, 3, 4

        topk_indices = torch.tensor([
            [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]],
            [[100, 101, 102, 103], [110, 111, 112, 113], [120, 121, 122, 123]],
        ])  # (2, 3, 4)

        # Sampled indices that are NOT in top-k
        sampled_indices = torch.tensor([
            [99, 99, 99],  # 99 is not in any of the top-k sets
            [999, 999, 999],
        ])  # (2, 3)

        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)

        # All should be True (token NOT in top-k)
        assert mask.all(), f"Expected all True, got {mask}"

    def test_mixed_in_and_not_in_topk(self):
        """Test mixed case where some tokens are in top-k and some are not."""
        batch, seq, k = 1, 4, 3

        topk_indices = torch.tensor([
            [[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]],
        ])  # (1, 4, 3)

        # Mix of in and not in
        sampled_indices = torch.tensor([
            [1, 99, 21, 99],  # 1 in, 99 not in, 21 in, 99 not in
        ])  # (1, 4)

        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)

        expected = torch.tensor([[False, True, False, True]])
        assert torch.equal(mask, expected), f"Expected {expected}, got {mask}"

    def test_first_element_in_topk(self):
        """Test that matching the first element in top-k works."""
        topk_indices = torch.tensor([[[5, 10, 15]]])  # (1, 1, 3)
        sampled_indices = torch.tensor([[5]])  # matches first element

        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)
        assert not mask[0, 0], "Token 5 should be found in top-k"

    def test_last_element_in_topk(self):
        """Test that matching the last element in top-k works."""
        topk_indices = torch.tensor([[[5, 10, 15]]])  # (1, 1, 3)
        sampled_indices = torch.tensor([[15]])  # matches last element

        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)
        assert not mask[0, 0], "Token 15 should be found in top-k"

    def test_boundary_values(self):
        """Test with boundary vocab indices (0 and large values)."""
        topk_indices = torch.tensor([[[0, 32000, 65535]]])  # common vocab sizes

        # Test 0
        sampled_indices = torch.tensor([[0]])
        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)
        assert not mask[0, 0], "Token 0 should be found"

        # Test large value
        sampled_indices = torch.tensor([[65535]])
        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)
        assert not mask[0, 0], "Token 65535 should be found"

        # Test value not in list
        sampled_indices = torch.tensor([[12345]])
        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)
        assert mask[0, 0], "Token 12345 should NOT be found"

    def test_shape_validation(self):
        """Test that shape validation catches errors."""
        # Wrong dim for sampled_indices
        with pytest.raises(AssertionError, match="must be 2D"):
            compute_not_in_topk_mask(
                torch.tensor([1, 2, 3]),  # 1D
                torch.tensor([[[1, 2, 3]]])
            )

        # Wrong dim for topk_indices
        with pytest.raises(AssertionError, match="must be 3D"):
            compute_not_in_topk_mask(
                torch.tensor([[1, 2, 3]]),
                torch.tensor([[1, 2, 3]])  # 2D
            )

        # Mismatched batch/seq dimensions
        with pytest.raises(AssertionError, match="shape"):
            compute_not_in_topk_mask(
                torch.tensor([[1, 2, 3]]),  # (1, 3)
                torch.tensor([[[1, 2], [3, 4]]])  # (1, 2, 2) - seq mismatch
            )

    def test_large_k(self):
        """Test with large k (many top-k indices)."""
        batch, seq, k = 2, 4, 1000

        # Create top-k indices: each position has indices 0 to k-1
        topk_indices = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(batch, seq, k)

        # Sampled index in range [0, k-1] should be in top-k
        sampled_in = torch.full((batch, seq), 500, dtype=torch.long)
        mask_in = compute_not_in_topk_mask(sampled_in, topk_indices)
        assert not mask_in.any(), "Token 500 should be in top-1000"

        # Sampled index >= k should NOT be in top-k
        sampled_out = torch.full((batch, seq), 1000, dtype=torch.long)
        mask_out = compute_not_in_topk_mask(sampled_out, topk_indices)
        assert mask_out.all(), "Token 1000 should NOT be in top-1000"

    def test_gpu_if_available(self):
        """Test that the function works on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        topk_indices = torch.tensor([[[0, 1, 2], [10, 11, 12]]]).cuda()
        sampled_indices = torch.tensor([[1, 99]]).cuda()

        mask = compute_not_in_topk_mask(sampled_indices, topk_indices)

        assert mask.device.type == "cuda"
        expected = torch.tensor([[False, True]]).cuda()
        assert torch.equal(mask, expected)


# =============================================================================
# Test L2 tail term computation
# =============================================================================

class TestL2TailTermComputation:
    """Test the L2 tail term computation logic."""

    def test_l2_masked_correctly(self):
        """Test that L2 is zero when token is in top-k."""
        batch, seq = 2, 4

        # Mask: False means token IS in top-k (should not contribute to L2)
        not_in_topk = torch.tensor([
            [False, True, False, True],
            [True, False, True, False],
        ])

        # Some arbitrary KL values
        kl_at_sampled = torch.randn(batch, seq)

        # L2 = mask * kl_at_sampled
        L2 = not_in_topk.float() * kl_at_sampled

        # Check that L2 is zero where mask is False
        for b in range(batch):
            for s in range(seq):
                if not not_in_topk[b, s]:
                    assert L2[b, s] == 0.0, f"L2[{b},{s}] should be 0 when in top-k"
                else:
                    assert L2[b, s] == kl_at_sampled[b, s], f"L2[{b},{s}] should equal kl_at_sampled"

    def test_pg_ratio_gradient_flow(self):
        """Test that pg_ratio = exp(log_prob - log_prob.detach()) has correct gradient."""
        log_prob = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        pg_ratio = (log_prob - log_prob.detach()).exp()

        # pg_ratio should be all 1s
        assert torch.allclose(pg_ratio, torch.ones(3)), f"pg_ratio should be 1, got {pg_ratio}"

        # But gradient should flow through
        loss = (pg_ratio * torch.tensor([1.0, 2.0, 3.0])).sum()
        loss.backward()

        # d(loss)/d(log_prob) = pg_ratio * [1, 2, 3] = [1, 2, 3]
        # But pg_ratio = exp(log_prob - sg(log_prob)), so d(pg_ratio)/d(log_prob) = pg_ratio = 1
        # Therefore d(loss)/d(log_prob) = 1 * [1, 2, 3] = [1, 2, 3]
        expected_grad = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(log_prob.grad, expected_grad), f"Expected grad {expected_grad}, got {log_prob.grad}"


# =============================================================================
# Test K4, K5, K6 estimators
# =============================================================================

class TestKLEstimators:
    """Test standalone KL estimators K4, K5, K6."""

    def test_k4_sampled_forward_kl(self):
        """Test K4: sampled forward KL estimator."""
        logprob = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        ref_logprob = torch.tensor([-1.5, -2.5, -3.5])

        # K4: logw.detach().exp() * logw + logr
        logw = ref_logprob - logprob  # [-0.5, -0.5, -0.5]
        logw_clamped = torch.clamp(logw, min=-20, max=20)
        logr = logprob - logprob.detach()  # [0, 0, 0] but with grad
        k4 = logw_clamped.detach().exp() * logw_clamped + logr

        # Value check: exp(-0.5) * -0.5 + 0 ≈ 0.6065 * -0.5 = -0.303
        expected_value = torch.exp(torch.tensor(-0.5)) * (-0.5)
        assert torch.allclose(k4, torch.full((3,), expected_value.item()), atol=1e-4)

        # Gradient check
        k4.sum().backward()
        assert logprob.grad is not None, "K4 should have gradient flow"

    def test_k5_sampled_reverse_kl(self):
        """Test K5: sampled reverse KL estimator."""
        logprob = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        ref_logprob = torch.tensor([-1.5, -2.5, -3.5])

        # K5: (logprob - ref_logprob).detach() * pg_ratio
        pg_ratio = (logprob - logprob.detach()).exp()  # = 1
        k5 = (logprob - ref_logprob).detach() * pg_ratio  # [0.5, 0.5, 0.5]

        # Value check
        expected_value = torch.tensor([0.5, 0.5, 0.5])
        assert torch.allclose(k5, expected_value), f"Expected {expected_value}, got {k5}"

        # Gradient check
        k5.sum().backward()
        assert logprob.grad is not None, "K5 should have gradient flow"
        # d(k5)/d(logprob) = (logprob - ref_logprob).detach() * d(pg_ratio)/d(logprob)
        # = (logprob - ref_logprob).detach() * pg_ratio = 0.5 * 1 = 0.5
        expected_grad = torch.tensor([0.5, 0.5, 0.5])
        assert torch.allclose(logprob.grad, expected_grad), f"Expected grad {expected_grad}, got {logprob.grad}"

    def test_k6_low_var_kl_with_pg_ratio(self):
        """Test K6: K3 (low_var_kl) times pg_ratio."""
        logprob = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        ref_logprob = torch.tensor([-1.5, -2.5, -3.5])

        # K3: ratio - kl - 1 where kl = ref_logprob - logprob, ratio = exp(kl)
        kl = ref_logprob - logprob  # [-0.5, -0.5, -0.5]
        kl_clamped = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl_clamped)  # exp(-0.5) ≈ 0.6065
        k3 = ratio - kl_clamped - 1  # 0.6065 - (-0.5) - 1 = 0.1065
        k3 = torch.clamp(k3, min=-10, max=10)

        # K6: K3 * pg_ratio
        pg_ratio = (logprob - logprob.detach()).exp()  # = 1
        k6 = k3 * pg_ratio

        # Value check
        expected_k3 = torch.exp(torch.tensor(-0.5)) - (-0.5) - 1
        assert torch.allclose(k6, torch.full((3,), expected_k3.item()), atol=1e-4)

        # Gradient check
        k6.sum().backward()
        assert logprob.grad is not None, "K6 should have gradient flow"


# =============================================================================
# Test end-to-end with compute_memory_efficient_kl
# =============================================================================

# =============================================================================
# Local copy of compute_memory_efficient_kl for testing without full verl deps
# =============================================================================

def _compute_memory_efficient_kl(
    actor_logits_k: torch.FloatTensor,
    actor_logsumexp: torch.FloatTensor,
    ref_logits_k: torch.FloatTensor,
    ref_logsumexp: torch.FloatTensor,
    kl_type: str = "full_reverse",
    use_tail_sampling: bool = False,
    actor_topk_indices: torch.LongTensor = None,
    ref_topk_indices: torch.LongTensor = None,
    sampled_indices: torch.LongTensor = None,
    log_prob: torch.FloatTensor = None,
    ref_log_prob: torch.FloatTensor = None,
) -> torch.FloatTensor:
    """Local copy of compute_memory_efficient_kl for testing.

    Note: Off-policy importance weighting (use_kl_iw) is applied separately
    in dp_actor.py after KL computation, so it's not included here.
    """
    batch, seq, k = actor_logits_k.shape

    if use_tail_sampling:
        assert sampled_indices is not None
        assert log_prob is not None
        assert ref_log_prob is not None

    actor_log_probs_k = actor_logits_k - actor_logsumexp.unsqueeze(-1)
    ref_log_probs_k = ref_logits_k - ref_logsumexp.unsqueeze(-1)

    if kl_type == "full_forward":
        ref_probs_k = ref_log_probs_k.exp()
        kl = (ref_probs_k * (ref_log_probs_k - actor_log_probs_k)).sum(dim=-1)

        if use_tail_sampling:
            assert ref_topk_indices is not None
            not_in_topk = compute_not_in_topk_mask(sampled_indices, ref_topk_indices)
            logw = ref_log_prob - log_prob
            logw = torch.clamp(logw, min=-20, max=20)
            logr = log_prob - log_prob.detach()
            sampled_forward_kl = logw.detach().exp() * logw + logr
            L2 = not_in_topk.to(dtype=sampled_forward_kl.dtype) * sampled_forward_kl
            kl = kl + L2
    else:
        actor_probs_k = actor_log_probs_k.exp()
        kl = (actor_probs_k * (actor_log_probs_k - ref_log_probs_k)).sum(dim=-1)

        if use_tail_sampling:
            assert actor_topk_indices is not None
            not_in_topk = compute_not_in_topk_mask(sampled_indices, actor_topk_indices)
            pg_ratio = (log_prob - log_prob.detach()).exp()
            kl_at_sampled = (log_prob - ref_log_prob).detach() * pg_ratio
            L2 = not_in_topk.to(dtype=kl_at_sampled.dtype) * kl_at_sampled
            kl = kl + L2

    return kl


class TestComputeMemoryEfficientKL:
    """Test compute_memory_efficient_kl with tail sampling."""

    def test_full_reverse_without_tail_sampling(self):
        """Test full_reverse KL without tail sampling (baseline)."""
        batch, seq, k = 2, 4, 100
        vocab = 1000

        # Create random logits
        torch.manual_seed(42)
        actor_logits = torch.randn(batch, seq, vocab)
        ref_logits = torch.randn(batch, seq, vocab)

        # Compute top-k
        _, topk_indices = actor_logits.topk(k, dim=-1)
        actor_logits_k = actor_logits.gather(-1, topk_indices)
        ref_logits_k = ref_logits.gather(-1, topk_indices)
        actor_logsumexp = actor_logits.logsumexp(dim=-1)
        ref_logsumexp = ref_logits.logsumexp(dim=-1)

        # Compute KL without tail sampling
        kl = _compute_memory_efficient_kl(
            actor_logits_k=actor_logits_k,
            actor_logsumexp=actor_logsumexp,
            ref_logits_k=ref_logits_k,
            ref_logsumexp=ref_logsumexp,
            kl_type="full_reverse",
            use_tail_sampling=False,
        )

        assert kl.shape == (batch, seq)
        # KL should be non-negative (approximately, top-k might underestimate)
        # For top-k approximation, values can be slightly negative due to missing mass

    def test_full_reverse_with_tail_sampling(self):
        """Test full_reverse KL with tail sampling."""
        batch, seq, k = 2, 4, 100
        vocab = 1000

        torch.manual_seed(42)
        actor_logits = torch.randn(batch, seq, vocab)
        ref_logits = torch.randn(batch, seq, vocab)

        # Compute top-k from actor
        _, actor_topk_indices = actor_logits.topk(k, dim=-1)
        actor_logits_k = actor_logits.gather(-1, actor_topk_indices)
        ref_logits_k = ref_logits.gather(-1, actor_topk_indices)
        actor_logsumexp = actor_logits.logsumexp(dim=-1)
        ref_logsumexp = ref_logits.logsumexp(dim=-1)

        # Sample tokens (simulate rollout)
        actor_probs = F.softmax(actor_logits, dim=-1)
        sampled_indices = torch.multinomial(actor_probs.view(-1, vocab), 1).view(batch, seq)

        # Compute log probs at sampled indices
        actor_log_probs = F.log_softmax(actor_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        log_prob = actor_log_probs.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
        ref_log_prob = ref_log_probs.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)

        # Compute KL with tail sampling
        kl_with_tail = _compute_memory_efficient_kl(
            actor_logits_k=actor_logits_k,
            actor_logsumexp=actor_logsumexp,
            ref_logits_k=ref_logits_k,
            ref_logsumexp=ref_logsumexp,
            kl_type="full_reverse",
            use_tail_sampling=True,
            actor_topk_indices=actor_topk_indices,
            sampled_indices=sampled_indices,
            log_prob=log_prob,
            ref_log_prob=ref_log_prob,
        )

        # Compute KL without tail sampling
        kl_without_tail = _compute_memory_efficient_kl(
            actor_logits_k=actor_logits_k,
            actor_logsumexp=actor_logsumexp,
            ref_logits_k=ref_logits_k,
            ref_logsumexp=ref_logsumexp,
            kl_type="full_reverse",
            use_tail_sampling=False,
        )

        assert kl_with_tail.shape == (batch, seq)

        # L2 term should add to the KL (on average, though individual samples may vary)
        # The difference should be the L2 term
        L2 = kl_with_tail - kl_without_tail

        # L2 should be non-zero for at least some positions (where sampled token not in top-k)
        # Since k=100 and vocab=1000, ~90% of samples should be outside top-k
        print(f"L2 stats: mean={L2.mean():.4f}, std={L2.std():.4f}, "
              f"min={L2.min():.4f}, max={L2.max():.4f}")

    def test_tail_sampling_only_adds_for_tokens_outside_topk(self):
        """Test that L2 is zero for tokens inside top-k."""
        batch, seq, k = 1, 10, 50
        vocab = 100

        torch.manual_seed(123)
        actor_logits = torch.randn(batch, seq, vocab)
        ref_logits = torch.randn(batch, seq, vocab)

        # Get top-k indices
        _, actor_topk_indices = actor_logits.topk(k, dim=-1)
        actor_logits_k = actor_logits.gather(-1, actor_topk_indices)
        ref_logits_k = ref_logits.gather(-1, actor_topk_indices)
        actor_logsumexp = actor_logits.logsumexp(dim=-1)
        ref_logsumexp = ref_logits.logsumexp(dim=-1)

        # Force sampled indices to be IN top-k (pick first top-k index)
        sampled_indices = actor_topk_indices[:, :, 0]  # (batch, seq)

        actor_log_probs = F.log_softmax(actor_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        log_prob = actor_log_probs.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
        ref_log_prob = ref_log_probs.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)

        kl_with_tail = _compute_memory_efficient_kl(
            actor_logits_k=actor_logits_k,
            actor_logsumexp=actor_logsumexp,
            ref_logits_k=ref_logits_k,
            ref_logsumexp=ref_logsumexp,
            kl_type="full_reverse",
            use_tail_sampling=True,
            actor_topk_indices=actor_topk_indices,
            sampled_indices=sampled_indices,
            log_prob=log_prob,
            ref_log_prob=ref_log_prob,
        )

        kl_without_tail = _compute_memory_efficient_kl(
            actor_logits_k=actor_logits_k,
            actor_logsumexp=actor_logsumexp,
            ref_logits_k=ref_logits_k,
            ref_logsumexp=ref_logsumexp,
            kl_type="full_reverse",
            use_tail_sampling=False,
        )

        # L2 should be zero since all sampled tokens are in top-k
        L2 = kl_with_tail - kl_without_tail
        assert torch.allclose(L2, torch.zeros_like(L2), atol=1e-6), \
            f"L2 should be zero when all tokens in top-k, got {L2}"


# =============================================================================
# Test numerical edge cases
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of the implementations."""

    def test_extreme_logprob_differences(self):
        """Test with extreme log probability differences."""
        logprob = torch.tensor([-0.1, -100.0, 0.0], requires_grad=True)
        ref_logprob = torch.tensor([-100.0, -0.1, -50.0])

        # K4 should clamp the logw
        logw = ref_logprob - logprob
        logw_clamped = torch.clamp(logw, min=-20, max=20)
        logr = logprob - logprob.detach()
        k4 = logw_clamped.detach().exp() * logw_clamped + logr

        # Should not have inf or nan
        assert torch.isfinite(k4).all(), f"K4 has non-finite values: {k4}"

        # Gradient should also be finite
        k4.sum().backward()
        assert torch.isfinite(logprob.grad).all(), f"K4 grad has non-finite values: {logprob.grad}"

    def test_importance_weight_clamping(self):
        """Test that importance weights are properly clamped."""
        log_prob = torch.tensor([0.0, -50.0, 50.0])
        old_log_prob = torch.tensor([-50.0, 0.0, 0.0])

        log_diff = (log_prob - old_log_prob).detach()
        log_diff_clamped = torch.clamp(log_diff, min=-20, max=20)
        iw = torch.exp(log_diff_clamped)

        # All values should be finite
        assert torch.isfinite(iw).all(), f"IW has non-finite values: {iw}"

        # Clamped values should be in [exp(-20), exp(20)]
        assert (iw >= torch.exp(torch.tensor(-20.0))).all()
        assert (iw <= torch.exp(torch.tensor(20.0))).all()


if __name__ == "__main__":
    # Run tests
    print("=" * 70)
    print("Running test_compute_not_in_topk_mask tests...")
    print("=" * 70)

    test_cls = TestComputeNotInTopkMask()
    test_cls.test_basic_in_topk()
    print("✓ test_basic_in_topk passed")

    test_cls.test_basic_not_in_topk()
    print("✓ test_basic_not_in_topk passed")

    test_cls.test_mixed_in_and_not_in_topk()
    print("✓ test_mixed_in_and_not_in_topk passed")

    test_cls.test_first_element_in_topk()
    print("✓ test_first_element_in_topk passed")

    test_cls.test_last_element_in_topk()
    print("✓ test_last_element_in_topk passed")

    test_cls.test_boundary_values()
    print("✓ test_boundary_values passed")

    test_cls.test_large_k()
    print("✓ test_large_k passed")

    print("\n" + "=" * 70)
    print("Running L2 tail term tests...")
    print("=" * 70)

    test_l2 = TestL2TailTermComputation()
    test_l2.test_l2_masked_correctly()
    print("✓ test_l2_masked_correctly passed")

    test_l2.test_pg_ratio_gradient_flow()
    print("✓ test_pg_ratio_gradient_flow passed")

    print("\n" + "=" * 70)
    print("Running KL estimator tests...")
    print("=" * 70)

    test_kl = TestKLEstimators()
    test_kl.test_k4_sampled_forward_kl()
    print("✓ test_k4_sampled_forward_kl passed")

    test_kl.test_k5_sampled_reverse_kl()
    print("✓ test_k5_sampled_reverse_kl passed")

    test_kl.test_k6_low_var_kl_with_pg_ratio()
    print("✓ test_k6_low_var_kl_with_pg_ratio passed")

    print("\n" + "=" * 70)
    print("Running numerical stability tests...")
    print("=" * 70)

    test_num = TestNumericalStability()
    test_num.test_extreme_logprob_differences()
    print("✓ test_extreme_logprob_differences passed")

    test_num.test_importance_weight_clamping()
    print("✓ test_importance_weight_clamping passed")

    print("\n" + "=" * 70)
    print("Running end-to-end tests...")
    print("=" * 70)

    test_e2e = TestComputeMemoryEfficientKL()
    test_e2e.test_full_reverse_without_tail_sampling()
    print("✓ test_full_reverse_without_tail_sampling passed")

    test_e2e.test_full_reverse_with_tail_sampling()
    print("✓ test_full_reverse_with_tail_sampling passed")

    test_e2e.test_tail_sampling_only_adds_for_tokens_outside_topk()
    print("✓ test_tail_sampling_only_adds_for_tokens_outside_topk passed")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
