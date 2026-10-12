from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual
    before using it.
    """

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    ref_adjusted_margin = policy_margin - ref_margin

    logits = beta * ref_adjusted_margin

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "reference_margin_mean":
            ref_margin.detach().mean(),

        "reference_adjusted_margin_mean":
            ref_adjusted_margin.detach().mean(),

        "dpo_margin_mean":
            ref_adjusted_margin.detach().mean(),

        "preference_accuracy": (
            ref_adjusted_margin > 0
        ).float().mean().detach(),

        "policy_preference_accuracy": (
            policy_margin > 0
        ).float().mean().detach(),
    }
