"""Comprehensive unit and objective tests for Task 3 GRPO.

Verifies the 10 required algorithmic and structural properties:
1. per-group normalization is independent between groups
2. zero-variance groups are numerically safe
3. group IDs correctly isolate groups
4. GRPO loss is finite
5. truncation masking produces zero valid tokens for truncated sequences
6. non-truncated sequences retain their response mask
7. canonical normalization uses realized valid token count
8. Dr.-GRPO normalization uses max_completion_length
9. policy loss can produce gradients when valid tokens exist
10. policy loss is zero / no-gradient when every sequence is correctly masked
"""
from __future__ import annotations

import torch
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def test_1_per_group_normalization_independent():
    """1. Per-group normalization is independent between groups."""
    rewards = torch.tensor([1.0, 3.0, 10.0, 12.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    adv = group_relative_advantages(rewards, group_ids)

    # Group 0: mean=2, std=1 -> [-1, 1]
    assert abs(adv[0].item() - (-1.0)) < 1e-4, f"Failed: adv[0]={adv[0]}"
    assert abs(adv[1].item() - 1.0) < 1e-4, f"Failed: adv[1]={adv[1]}"
    # Group 1: mean=11, std=1 -> [-1, 1]
    assert abs(adv[2].item() - (-1.0)) < 1e-4, f"Failed: adv[2]={adv[2]}"
    assert abs(adv[3].item() - 1.0) < 1e-4, f"Failed: adv[3]={adv[3]}"
    print("PASS 1: per-group normalization is independent between groups")


def test_2_zero_variance_groups_safe():
    """2. Zero-variance groups are numerically safe (no NaNs or Infs)."""
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
    group_ids = torch.tensor([0, 0, 0, 0])
    adv = group_relative_advantages(rewards, group_ids)
    assert torch.isfinite(adv).all(), f"Failed: adv contains non-finite values: {adv}"
    assert (adv == 0.0).all(), f"Expected zeros for zero-variance group, got {adv}"
    print("PASS 2: zero-variance groups are numerically safe")


def test_3_group_ids_isolate_groups():
    """3. Group IDs correctly isolate groups (different shifts do not affect other groups)."""
    # Group 0 has rewards [0, 2], Group 1 has rewards [100, 102]
    # Changing Group 1 to [500, 502] should NOT change Group 0's advantages
    rewards_a = torch.tensor([0.0, 2.0, 100.0, 102.0])
    rewards_b = torch.tensor([0.0, 2.0, 500.0, 502.0])
    gids = torch.tensor([0, 0, 1, 1])

    adv_a = group_relative_advantages(rewards_a, gids)
    adv_b = group_relative_advantages(rewards_b, gids)

    assert torch.allclose(adv_a[:2], adv_b[:2], atol=1e-5), "Failed: group 1 shift contaminated group 0"
    print("PASS 3: group IDs correctly isolate groups")


def test_4_grpo_loss_finite():
    """4. GRPO loss is finite."""
    B, L = 4, 16
    nlp = torch.randn(B, L, requires_grad=True) * 0.1
    olp = nlp.detach() + torch.randn(B, L) * 0.01
    rlp = nlp.detach() + torch.randn(B, L) * 0.01
    adv = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    mask = torch.ones(B, L)

    loss, stats = grpo_policy_loss(nlp, olp, adv, mask, rlp, eps=0.2, beta=0.1, loss_type="grpo")
    assert torch.isfinite(loss), f"Failed: GRPO loss is not finite: {loss}"
    print("PASS 4: GRPO loss is finite")


def test_5_truncation_masking_zeroes_truncated():
    """5. Truncation masking produces zero valid tokens for truncated sequences."""
    mask = torch.ones(3, 10)
    truncated = [True, False, True]
    masked = mask_truncated_sequences(mask, truncated)

    assert (masked[0] == 0.0).all(), "Sequence 0 was truncated but not zeroed"
    assert (masked[2] == 0.0).all(), "Sequence 2 was truncated but not zeroed"
    print("PASS 5: truncation masking produces zero valid tokens for truncated sequences")


def test_6_non_truncated_retains_mask():
    """6. Non-truncated sequences retain their response mask."""
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    truncated = [False, True]
    masked = mask_truncated_sequences(mask, truncated)

    assert torch.equal(masked[0], mask[0]), "Non-truncated sequence mask was modified"
    print("PASS 6: non-truncated sequences retain their response mask")


def test_7_canonical_normalization_realized_tokens():
    """7. Canonical normalization uses realized valid token count."""
    L = 10
    nlp = torch.zeros(1, L, requires_grad=True)
    olp = torch.zeros(1, L)
    rlp = torch.zeros(1, L)
    adv = torch.tensor([1.0])
    mask = torch.zeros(1, L)
    mask[0, :5] = 1.0  # Realized valid tokens = 5

    loss, stats = grpo_policy_loss(nlp, olp, adv, mask, rlp, eps=0.2, beta=0.0, loss_type="grpo")
    # Objective per token = 1.0 * 1.0 = 1.0 for each of the 5 tokens
    # sum = 5.0, denom = 5.0 -> per_sequence = 1.0 -> policy_term = -1.0 -> loss = -1.0
    expected = -1.0
    assert abs(loss.item() - expected) < 1e-4, f"Expected {expected}, got {loss.item()}"
    print("PASS 7: canonical normalization uses realized valid token count")


def test_8_dr_grpo_normalization_max_length():
    """8. Dr.-GRPO normalization uses max_completion_length."""
    L = 10
    max_len = 20
    nlp = torch.zeros(1, L, requires_grad=True)
    olp = torch.zeros(1, L)
    rlp = torch.zeros(1, L)
    adv = torch.tensor([1.0])
    mask = torch.zeros(1, L)
    mask[0, :5] = 1.0  # Realized tokens = 5

    loss, stats = grpo_policy_loss(
        nlp, olp, adv, mask, rlp, eps=0.2, beta=0.0, loss_type="dr_grpo", max_completion_length=max_len
    )
    # Objective per token = 1.0, sum over 5 tokens = 5.0
    # denom = 20.0 -> per_sequence = 5.0 / 20.0 = 0.25 -> policy_term = -0.25 -> loss = -0.25
    expected = -0.25
    assert abs(loss.item() - expected) < 1e-4, f"Expected {expected}, got {loss.item()}"
    print("PASS 8: Dr.-GRPO normalization uses max_completion_length")


def test_9_policy_loss_produces_gradients():
    """9. Policy loss can produce gradients when valid tokens exist."""
    B, L = 2, 8
    nlp = torch.randn(B, L, requires_grad=True)
    olp = nlp.detach().clone()
    rlp = nlp.detach().clone()
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(B, L)

    loss, _ = grpo_policy_loss(nlp, olp, adv, mask, rlp, eps=0.2, beta=0.1, loss_type="grpo")
    loss.backward()

    assert nlp.grad is not None, "Gradient is None"
    assert (nlp.grad != 0.0).any(), "Gradients are all zero despite valid tokens"
    print("PASS 9: policy loss can produce gradients when valid tokens exist")


def test_10_policy_loss_zero_when_all_masked():
    """10. Policy loss is zero/no-gradient when every sequence is correctly masked."""
    B, L = 2, 8
    nlp = torch.randn(B, L, requires_grad=True)
    olp = nlp.detach().clone()
    rlp = nlp.detach().clone()
    adv = torch.tensor([1.0, -1.0])
    # Every sequence is masked to 0
    mask = torch.zeros(B, L)

    loss, _ = grpo_policy_loss(nlp, olp, adv, mask, rlp, eps=0.2, beta=0.1, loss_type="grpo")
    assert abs(loss.item()) < 1e-6, f"Expected zero loss when all masked, got {loss.item()}"

    loss.backward()
    # Gradient should be None or all zeros
    if nlp.grad is not None:
        assert (nlp.grad == 0.0).all(), f"Expected zero gradient when all masked, got {nlp.grad}"
    print("PASS 10: policy loss is zero/no-gradient when every sequence is correctly masked")


def run_all_tests():
    print("=" * 60)
    print("RUNNING ALL 10 TASK 3 GRPO OBJECTIVE TESTS")
    print("=" * 60)
    test_1_per_group_normalization_independent()
    test_2_zero_variance_groups_safe()
    test_3_group_ids_isolate_groups()
    test_4_grpo_loss_finite()
    test_5_truncation_masking_zeroes_truncated()
    test_6_non_truncated_retains_mask()
    test_7_canonical_normalization_realized_tokens()
    test_8_dr_grpo_normalization_max_length()
    test_9_policy_loss_produces_gradients()
    test_10_policy_loss_zero_when_all_masked()
    print("=" * 60)
    print("ALL 10 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
