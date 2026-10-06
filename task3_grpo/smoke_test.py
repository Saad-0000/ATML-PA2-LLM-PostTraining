"""Smoke-test for task3_grpo.grpo — run before full experiments."""
from __future__ import annotations
import torch
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss

def test_per_group_normalization():
    """Two prompt groups must be normalized independently."""
    rewards   = torch.tensor([1.0, 3.0, 10.0, 12.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    adv = group_relative_advantages(rewards, group_ids)

    # Group 0: rewards [1,3], mean=2, std=1 → adv=[-1,+1]
    assert abs(adv[0].item() - (-1.0)) < 1e-5, f"group0[0] got {adv[0]}"
    assert abs(adv[1].item() -   1.0 ) < 1e-5, f"group0[1] got {adv[1]}"
    # Group 1: rewards [10,12], mean=11, std=1 → adv=[-1,+1]
    assert abs(adv[2].item() - (-1.0)) < 1e-5, f"group1[0] got {adv[2]}"
    assert abs(adv[3].item() -   1.0 ) < 1e-5, f"group1[1] got {adv[3]}"
    print("PASS: per-group normalization is independent")

def test_zero_variance_group():
    """Zero-variance group must not produce NaN."""
    rewards   = torch.tensor([5.0, 5.0])
    group_ids = torch.tensor([0, 0])
    adv = group_relative_advantages(rewards, group_ids)
    assert torch.isfinite(adv).all(), f"Got NaN/Inf for zero-variance group: {adv}"
    print("PASS: zero-variance group is safe")

def test_global_vs_group_differ():
    """Prove the bug: global normalization gives DIFFERENT results than per-group."""
    rewards   = torch.tensor([0.0, 10.0, 0.0, 10.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    adv_group  = group_relative_advantages(rewards, group_ids)
    # Global (buggy): mean=5, std=5 → adv=[-1,+1,-1,+1]
    # Per-group: each group [0,10] → adv=[-1,+1] (same result here, but the
    # mechanism is per-group so mixing won't occur).
    # Test a case where groups differ to prove mixing is absent:
    rewards2   = torch.tensor([0.0, 10.0, 100.0, 110.0])
    group_ids2 = torch.tensor([0, 0, 1, 1])
    adv_pg = group_relative_advantages(rewards2, group_ids2)
    # Per-group: group0 mean=5,std=5; group1 mean=105,std=5 → adv same pattern
    assert abs(adv_pg[0].item() - (-1.0)) < 1e-4
    assert abs(adv_pg[1].item() -   1.0 ) < 1e-4
    assert abs(adv_pg[2].item() - (-1.0)) < 1e-4
    assert abs(adv_pg[3].item() -   1.0 ) < 1e-4
    print("PASS: per-group isolation verified")

def test_grpo_loss_finite():
    """grpo_policy_loss returns finite values."""
    B, L = 4, 20
    new_lp = torch.randn(B, L) * 0.1
    old_lp = new_lp + torch.randn(B, L) * 0.01
    ref_lp = new_lp + torch.randn(B, L) * 0.01
    adv    = torch.randn(B)
    mask   = torch.ones(B, L)
    loss, stats = grpo_policy_loss(new_lp, old_lp, adv, mask, ref_lp, eps=0.2, beta=0.1, loss_type="grpo")
    assert torch.isfinite(loss), f"grpo loss non-finite: {loss}"
    loss_dr, _ = grpo_policy_loss(new_lp, old_lp, adv, mask, ref_lp, eps=0.2, beta=0.1, loss_type="dr_grpo", max_completion_length=20)
    assert torch.isfinite(loss_dr), f"dr_grpo loss non-finite: {loss_dr}"
    print("PASS: grpo_policy_loss produces finite values")

if __name__ == "__main__":
    test_per_group_normalization()
    test_zero_variance_group()
    test_global_vs_group_differ()
    test_grpo_loss_finite()
    print("\nAll smoke tests passed.")
