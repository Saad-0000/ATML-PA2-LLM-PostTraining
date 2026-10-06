from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)
from torch.optim import AdamW


# ---------------------------------------------------------------------------
# Helper: generate K completions for ONE prompt (no grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_k(policy, tokenizer, prompt_ids, K, max_new, gen_cfg, device):
    inp  = torch.tensor([prompt_ids], device=device)
    attn = torch.ones_like(inp)
    results = []
    for _ in range(K):
        out = policy.generate(
            inp,
            attention_mask=attn,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=gen_cfg.get("do_sample", True),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
        )
        results.append(out[0][len(prompt_ids):].cpu())
    return results


@torch.no_grad()
def get_logprobs(policy, full_ids, attn, resp_mask):
    logits = policy(input_ids=full_ids, attention_mask=attn).logits.float()
    labels = full_ids[:, 1:]
    lp = logits[:, :-1, :].log_softmax(-1)
    per_tok = torch.gather(lp, 2, labels.unsqueeze(2)).squeeze(2)
    mask = resp_mask[:, 1:]
    return per_tok, mask


# ---------------------------------------------------------------------------
# Run one short fork for a given loss_type
# ---------------------------------------------------------------------------

def run_fork(
    cfg: dict,
    loss_type: str,
    prompts: list[dict],
    tokenizer,
    policy_state_dict: dict,  # initial weights shared between forks
    reward_model,
    r_tok,
    fork_updates: int,
    device,
) -> dict[str, Any]:
    """Run a short fork and collect per-response length-conditioned stats."""
    # Reload a fresh copy of the policy from the initial state
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))

    K           = int(cfg.get("num_generations", 4))
    max_prompt  = int(cfg.get("max_prompt_length", 256))
    max_comp    = int(cfg.get("max_completion_length", 256))
    clip_eps    = float(cfg["clip_epsilon"])
    kl_beta     = float(cfg["kl_beta"])
    max_grad_n  = float(cfg.get("max_grad_norm", 1.0))
    mask_trunc  = bool(cfg.get("mask_truncated_completions", True))
    reward_max  = int(cfg.get("reward_max_length", 1280))
    gen_cfg     = cfg.get("generation", {})

    per_response_records: list[dict] = []
    update_records: list[dict] = []

    for upd in range(fork_updates):
        policy.eval()
        prompt_idx = upd % len(prompts)
        row = prompts[prompt_idx]
        msgs = prompt_messages(row)
        prompt_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[:max_prompt]

        completions = generate_k(policy, tokenizer, prompt_ids, K, max_comp, gen_cfg, device)

        rewards_list: list[float] = []
        seq_data: list[dict] = []

        for i, comp_ids in enumerate(completions):
            truncated = (
                len(comp_ids) >= max_comp
                and (tokenizer.eos_token_id is None
                     or comp_ids[-1].item() != tokenizer.eos_token_id)
            )
            full_ids = torch.tensor(
                prompt_ids + comp_ids.tolist(), dtype=torch.long, device=device
            ).unsqueeze(0)
            attn      = torch.ones_like(full_ids)
            resp_mask = torch.zeros_like(full_ids, dtype=torch.float32)
            resp_mask[0, len(prompt_ids):] = 1.0

            full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
            rm_in = r_tok(
                full_text, return_tensors="pt", truncation=True, max_length=reward_max
            ).to(reward_model.device)
            with torch.no_grad():
                rew = reward_model(**rm_in).logits[0, 0].item()
            rewards_list.append(rew)

            with torch.no_grad():
                old_lp, tok_mask = get_logprobs(policy, full_ids, attn, resp_mask)
                with reference_mode(policy):
                    ref_lp, _ = get_logprobs(policy, full_ids, attn, resp_mask)

            resp_len = int(tok_mask.sum().item())
            seq_data.append({
                "full_ids":  full_ids,
                "attn":      attn,
                "resp_mask": resp_mask,
                "tok_mask":  tok_mask,
                "old_lp":    old_lp.detach(),
                "ref_lp":    ref_lp.detach(),
                "reward":    rew,
                "resp_len":  resp_len,
                "truncated": truncated,
                "resp_text": tokenizer.decode(comp_ids, skip_special_tokens=True),
            })

        rewards_t  = torch.tensor(rewards_list, dtype=torch.float32)
        group_ids  = torch.zeros(K, dtype=torch.long)
        seq_adv    = group_relative_advantages(rewards_t, group_ids)

        # Forward + backward
        policy.train()
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        seq_objective_mags: list[float] = []

        for i, sd in enumerate(seq_data):
            cur_lp, tok_mask = get_logprobs(policy, sd["full_ids"], sd["attn"], sd["resp_mask"])
            cur_2d  = cur_lp.unsqueeze(0)
            old_2d  = sd["old_lp"].unsqueeze(0)
            ref_2d  = sd["ref_lp"].unsqueeze(0)
            mask_2d = tok_mask.unsqueeze(0)
            if mask_trunc and sd["truncated"]:
                mask_2d = mask_truncated_sequences(mask_2d, [True])

            adv_i = seq_adv[i].unsqueeze(0)
            loss_i, stats_i = grpo_policy_loss(
                new_logp=cur_2d,
                old_logp=old_2d,
                seq_adv=adv_i,
                token_mask=mask_2d,
                ref_logp=ref_2d,
                eps=clip_eps,
                beta=kl_beta,
                loss_type=loss_type,
                max_completion_length=max_comp,
            )
            batch_loss = batch_loss + loss_i / K
            # Objective magnitude = |policy_term| (absolute contribution per sequence)
            seq_objective_mags.append(abs(stats_i["policy_term"].item()))

            per_response_records.append({
                "update":           upd + 1,
                "loss_type":        loss_type,
                "seq_idx":          i,
                "resp_len":         sd["resp_len"],
                "reward":           sd["reward"],
                "advantage":        float(seq_adv[i].item()),
                "policy_term_mag":  abs(stats_i["policy_term"].item()),
                "kl":               stats_i["sampled_kl"].item(),
                "truncated":        sd["truncated"],
                "resp_text_prefix": sd["resp_text"][:100],
            })

        if torch.isfinite(batch_loss):
            batch_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_n).item()
            optimizer.step()
        else:
            gn = float("nan")

        update_records.append({
            "update":              upd + 1,
            "loss_type":           loss_type,
            "mean_reward":         float(rewards_t.mean().item()),
            "mean_objective_mag":  float(np.mean(seq_objective_mags)),
            "grad_norm":           gn,
        })
        print(
            f"  [{loss_type}] Update {upd+1}/{fork_updates} | "
            f"Reward={float(rewards_t.mean()):.4f} | Loss={batch_loss.item():.6f} | "
            f"GradNorm={gn:.4f}"
        )

    del policy, optimizer
    clear_gpu()

    return {
        "loss_type":            loss_type,
        "updates":              update_records,
        "per_response_records": per_response_records,
    }


# ---------------------------------------------------------------------------
# Length-conditioned analysis
# ---------------------------------------------------------------------------

def length_conditioned_analysis(
    records: list[dict],
    short_threshold: float = 128.0,
) -> dict[str, Any]:
    """Compare objective magnitude and reward across short vs long responses."""
    short = [r for r in records if r["resp_len"] <= short_threshold]
    long  = [r for r in records if r["resp_len"] >  short_threshold]

    def agg(group):
        if not group:
            return {}
        return {
            "n":                   len(group),
            "mean_resp_len":       float(np.mean([r["resp_len"] for r in group])),
            "mean_reward":         float(np.mean([r["reward"] for r in group])),
            "mean_advantage_abs":  float(np.mean([abs(r["advantage"]) for r in group])),
            "mean_policy_term_mag": float(np.mean([r["policy_term_mag"] for r in group])),
            "mean_kl":             float(np.mean([r["kl"] for r in group])),
        }

    return {
        "short_responses": agg(short),
        "long_responses":  agg(long),
        "length_threshold_tokens": short_threshold,
        "note": (
            "policy_term_mag is the per-sequence |policy_term| before KL. "
            "Canonical GRPO divides by realized length; Dr. GRPO divides by "
            "max_completion_length. Short responses get higher per-token weight "
            "under canonical normalization."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",        default="configs/grpo.yaml")
    ap.add_argument("--fork-updates",  type=int)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    fork_updates = args.fork_updates or int(cfg.get("fork_updates", 8))

    res_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    res_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg["base_model"])
    prompts   = read_jsonl(cfg["paths"]["rl_prompt_train"])
    reward_model, r_tok = load_reward_model(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compare normalization | fork_updates={fork_updates} | device={device}")
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    all_results: dict[str, Any] = {}
    all_per_resp: dict[str, list[dict]] = {}

    for loss_type in ["grpo", "dr_grpo"]:
        print(f"\n=== Fork: {loss_type} ===")
        result = run_fork(
            cfg=cfg,
            loss_type=loss_type,
            prompts=prompts,
            tokenizer=tokenizer,
            policy_state_dict=None,  # each fork loads fresh from midpoint
            reward_model=reward_model,
            r_tok=r_tok,
            fork_updates=fork_updates,
            device=device,
        )
        all_results[loss_type] = result["updates"]
        all_per_resp[loss_type] = result["per_response_records"]

    # ---- length-conditioned comparison ----------------------------------
    lc_grpo    = length_conditioned_analysis(all_per_resp["grpo"])
    lc_dr_grpo = length_conditioned_analysis(all_per_resp["dr_grpo"])

    # ---- qualitative length examples ------------------------------------
    def find_length_examples(records, loss_type):
        short_ex = min(records, key=lambda r: r["resp_len"], default=None)
        long_ex  = max(records, key=lambda r: r["resp_len"], default=None)
        return {"shortest": short_ex, "longest": long_ex}

    qual = {
        "grpo":    find_length_examples(all_per_resp["grpo"],    "grpo"),
        "dr_grpo": find_length_examples(all_per_resp["dr_grpo"], "dr_grpo"),
    }

    output = {
        "config": {
            "fork_updates":          fork_updates,
            "max_completion_length": int(cfg.get("max_completion_length", 256)),
            "K":                     int(cfg.get("num_generations", 4)),
            "clip_epsilon":          float(cfg["clip_epsilon"]),
            "kl_beta":               float(cfg["kl_beta"]),
            "seed":                  int(cfg["seed"]),
            "midpoint_checkpoint":   cfg["paths"]["grpo_midpoint_policy"],
        },
        "update_history": all_results,
        "length_conditioned": {
            "grpo":    lc_grpo,
            "dr_grpo": lc_dr_grpo,
        },
        "per_response_records": all_per_resp,
        "qualitative_length_examples": qual,
        "interpretation": {
            "canonical_grpo": (
                "Divides each sequence's clipped objective by its REALIZED response length. "
                "Short responses get higher per-token weight, potentially biasing the gradient "
                "toward brevity."
            ),
            "dr_grpo": (
                "Divides by fixed max_completion_length. All responses get the same denominator "
                "regardless of realized length, giving shorter responses proportionally less "
                "gradient influence relative to their length."
            ),
        },
    }

    out_path = res_dir / "normalization.json"
    save_json(out_path, output)
    print(f"\nNormalization comparison saved to {out_path}")

    # Print summary table
    print("\n--- Length-conditioned summary ---")
    for lt in ["grpo", "dr_grpo"]:
        lc = output["length_conditioned"][lt]
        s  = lc.get("short_responses", {})
        lo = lc.get("long_responses",  {})
        print(
            f"[{lt}] Short (≤{lc['length_threshold_tokens']}): "
            f"N={s.get('n',0)} mean_obj={s.get('mean_policy_term_mag', float('nan')):.4f} | "
            f"Long: N={lo.get('n',0)} mean_obj={lo.get('mean_policy_term_mag', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
