from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sampled_kl
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


def encode_prompt_ids(tokenizer, msgs, max_len: int | None = None) -> list[int]:
    """Encode chat messages into a list of integer token IDs using return_tensors='pt'."""
    tensor = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(tensor, "input_ids"):
        ids = tensor.input_ids[0].tolist()
    elif hasattr(tensor, "tolist"):
        ids = tensor[0].tolist()
    elif isinstance(tensor, list) and len(tensor) > 0 and hasattr(tensor[0], "ids"):
        ids = list(tensor[0].ids)
    elif isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], list):
        ids = list(tensor[0])
    else:
        ids = list(tensor)

    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
    return ids


@torch.no_grad()
def generate_k(policy, tokenizer, prompt_ids: list[int], K: int, max_new: int, gen_cfg: dict, device):
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
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


@torch.no_grad()
def evaluate_heldout_subset(policy, ref_policy, reward_bundle, tokenizer, eval_rows, cfg, n_eval: int = 10):
    """Evaluate held-out reward, KL, and length on a deterministic subset of eval prompts."""
    reward_model, r_tok = reward_bundle
    device = next(policy.parameters()).device
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen = int(cfg.get("max_completion_length", 512))
    reward_max = int(cfg.get("reward_max_length", 1280))

    rewards, lengths, kls = [], [], []

    for row in eval_rows[:n_eval]:
        msgs = prompt_messages(row)
        prompt_ids = encode_prompt_ids(tokenizer, msgs, max_prompt)

        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)
        out = policy.generate(
            inp,
            attention_mask=attn,
            max_new_tokens=max_gen,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        gen_ids = out[0][len(prompt_ids):]
        if len(gen_ids) == 0:
            continue
        lengths.append(len(gen_ids))

        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        rm_in = r_tok(full_text, return_tensors="pt", truncation=True, max_length=reward_max).to(reward_model.device)
        score = reward_model(**rm_in).logits[0, 0].item()
        rewards.append(score)

        resp_mask = torch.zeros_like(out[:, 1:], dtype=torch.float32)
        resp_mask[0, len(prompt_ids) - 1 :] = 1.0

        pol_lp, _ = get_logprobs(policy, out, torch.ones_like(out), torch.ones_like(out))
        with reference_mode(policy):
            ref_lp, _ = get_logprobs(policy, out, torch.ones_like(out), torch.ones_like(out))

        kl_val = sampled_kl(pol_lp, ref_lp, resp_mask).item()
        kls.append(kl_val)

    return {
        "heldout_reward": float(np.mean(rewards)) if rewards else 0.0,
        "heldout_kl": float(np.mean(kls)) if kls else 0.0,
        "heldout_length": float(np.mean(lengths)) if lengths else 0.0,
        "eval_count": len(rewards),
    }


def run_normalization_fork(
    cfg: dict,
    loss_type: str,
    prompts: list[dict],
    tokenizer,
    reward_model,
    r_tok,
    fork_updates: int,
    device,
) -> dict[str, Any]:
    midpoint_path = cfg["paths"]["grpo_midpoint_policy"]
    p = repo_path(midpoint_path)
    has_adapter = False
    if (p / "adapter_config.json").exists():
        try:
            with open(p / "adapter_config.json", encoding="utf-8") as f:
                c = f.read(20)
                if c.strip().startswith("{"):
                    has_adapter = True
        except Exception:
            pass

    if has_adapter:
        policy = load_policy(cfg, adapter_path=midpoint_path, trainable=True)
    else:
        policy = load_policy(cfg, adapter_path=None, trainable=True, fresh_lora=True)

    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))

    K          = int(cfg.get("num_generations", 4))
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_comp   = int(cfg.get("max_completion_length", 512))
    clip_eps   = float(cfg["clip_epsilon"])
    kl_beta    = float(cfg["kl_beta"])
    max_grad_n = float(cfg.get("max_grad_norm", 1.0))
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    reward_max = int(cfg.get("reward_max_length", 1280))
    gen_cfg    = cfg.get("generation", {})

    per_response_records = []
    update_records = []

    for upd in range(fork_updates):
        policy.eval()
        prompt_idx = upd % len(prompts)
        row = prompts[prompt_idx]
        msgs = prompt_messages(row)
        p_id = str(row.get("prompt_id", f"prompt_{prompt_idx}"))
        prompt_ids = encode_prompt_ids(tokenizer, msgs, max_prompt)

        completions = generate_k(policy, tokenizer, prompt_ids, K, max_comp, gen_cfg, device)

        rewards_list = []
        seq_data = []

        for comp_ids in completions:
            truncated = (
                len(comp_ids) >= max_comp
                and (tokenizer.eos_token_id is None or comp_ids[-1].item() != tokenizer.eos_token_id)
            )
            full_ids = torch.tensor(prompt_ids + comp_ids.tolist(), dtype=torch.long, device=device).unsqueeze(0)
            attn = torch.ones_like(full_ids)
            resp_mask = torch.zeros_like(full_ids, dtype=torch.float32)
            resp_mask[0, len(prompt_ids):] = 1.0

            full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
            rm_in = r_tok(full_text, return_tensors="pt", truncation=True, max_length=reward_max).to(reward_model.device)
            with torch.no_grad():
                rew = reward_model(**rm_in).logits[0, 0].item()
            rewards_list.append(rew)

            with torch.no_grad():
                old_lp, tok_mask = get_logprobs(policy, full_ids, attn, resp_mask)
                with reference_mode(policy):
                    ref_lp, _ = get_logprobs(policy, full_ids, attn, resp_mask)

            resp_len = int(tok_mask.sum().item())
            seq_data.append({
                "full_ids": full_ids,
                "attn": attn,
                "resp_mask": resp_mask,
                "tok_mask": tok_mask,
                "old_lp": old_lp.detach(),
                "ref_lp": ref_lp.detach(),
                "reward": rew,
                "resp_len": resp_len,
                "truncated": truncated,
                "resp_text": tokenizer.decode(comp_ids, skip_special_tokens=True),
            })

        rewards_t = torch.tensor(rewards_list, dtype=torch.float32)
        group_ids = torch.zeros(K, dtype=torch.long)
        seq_adv = group_relative_advantages(rewards_t, group_ids)

        policy.train()
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        seq_objective_mags = []

        for i, sd in enumerate(seq_data):
            cur_lp, tok_mask = get_logprobs(policy, sd["full_ids"], sd["attn"], sd["resp_mask"])
            cur_2d = cur_lp.unsqueeze(0)
            old_2d = sd["old_lp"].unsqueeze(0)
            ref_2d = sd["ref_lp"].unsqueeze(0)
            mask_2d = tok_mask.unsqueeze(0)
            if mask_trunc and sd["truncated"]:
                mask_2d = mask_truncated_sequences(mask_2d, [True])

            adv_i = seq_adv[i].unsqueeze(0).to(cur_2d.device)
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
            policy_mag = abs(stats_i["policy_term"].item())
            seq_objective_mags.append(policy_mag)

            per_response_records.append({
                "update": upd + 1,
                "prompt_id": p_id,
                "loss_type": loss_type,
                "seq_idx": i,
                "resp_len": sd["resp_len"],
                "reward": sd["reward"],
                "advantage": float(seq_adv[i].item()),
                "policy_term_magnitude": policy_mag,
                "per_token_contribution": float(policy_mag / max(sd["resp_len"], 1)),
                "kl": stats_i["sampled_kl"].item(),
                "truncated": sd["truncated"],
                "resp_text": sd["resp_text"][:200],
            })

        if torch.isfinite(batch_loss) and batch_loss.requires_grad and batch_loss.item() != 0.0:
            batch_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_n).item()
            optimizer.step()
        else:
            gn = 0.0

        update_records.append({
            "update": upd + 1,
            "loss_type": loss_type,
            "mean_reward": float(rewards_t.mean().item()),
            "mean_objective_magnitude": float(np.mean(seq_objective_mags)),
            "grad_norm": gn,
        })
        print(
            f"  [{loss_type}] Update {upd+1}/{fork_updates} | "
            f"Reward={float(rewards_t.mean()):.4f} | Loss={batch_loss.item():.6f} | GradNorm={gn:.4f}"
        )

    # Evaluate held-out metrics
    eval_rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    heldout_stats = evaluate_heldout_subset(policy, None, (reward_model, r_tok), tokenizer, eval_rows, cfg)
    print(f"  [{loss_type}] Held-out: reward={heldout_stats['heldout_reward']:.4f} | KL={heldout_stats['heldout_kl']:.4f} | len={heldout_stats['heldout_length']:.1f}")

    del policy, optimizer
    clear_gpu()

    return {
        "loss_type": loss_type,
        "heldout": heldout_stats,
        "updates": update_records,
        "per_response_records": per_response_records,
    }


def analyze_length_bins(records: list[dict], threshold: float = 200.0) -> dict[str, Any]:
    short = [r for r in records if r["resp_len"] <= threshold]
    long  = [r for r in records if r["resp_len"] > threshold]

    def summarize(subset):
        if not subset:
            return {"count": 0, "mean_length": 0.0, "mean_policy_mag": 0.0, "mean_per_token_contrib": 0.0}
        return {
            "count": len(subset),
            "mean_length": float(np.mean([r["resp_len"] for r in subset])),
            "mean_policy_mag": float(np.mean([r["policy_term_magnitude"] for r in subset])),
            "mean_per_token_contrib": float(np.mean([r["per_token_contribution"] for r in subset])),
            "mean_reward": float(np.mean([r["reward"] for r in subset])),
        }

    return {
        "length_threshold": threshold,
        "binning_rule": f"Completions with realized length <= {threshold} tokens are 'short'; > {threshold} tokens are 'long'.",
        "short_completions": summarize(short),
        "long_completions": summarize(long),
    }


def run_normalization_comparison(config_path: str = "configs/grpo.yaml"):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    fork_updates = int(cfg.get("fork_updates", 8))
    res_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / "normalization.json"
    qual_file = res_dir / "qualitative_examples.jsonl"

    tokenizer = load_tokenizer(cfg["base_model"])
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    reward_model, r_tok = load_reward_model(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("STARTING LENGTH-NORMALIZATION COMPARISON: CANONICAL VS DR. GRPO")
    print(f"Fixed: seed={cfg['seed']}, K={cfg['num_generations']}, beta={cfg['kl_beta']}, eps={cfg['clip_epsilon']}")
    print(f"Fork updates = {fork_updates}")
    print("=" * 60)

    fork_results = {}
    for loss_type in ["grpo", "dr_grpo"]:
        print(f"\n--- Running Fork: {loss_type} ---")
        res = run_normalization_fork(
            cfg=cfg,
            loss_type=loss_type,
            prompts=prompts,
            tokenizer=tokenizer,
            reward_model=reward_model,
            r_tok=r_tok,
            fork_updates=fork_updates,
            device=device,
        )
        fork_results[loss_type] = res

    # Length-conditioned analysis
    length_analysis = {
        "canonical_grpo": analyze_length_bins(fork_results["grpo"]["per_response_records"], threshold=200.0),
        "dr_grpo": analyze_length_bins(fork_results["dr_grpo"]["per_response_records"], threshold=200.0),
    }

    # Qualitative examples: shortest vs longest response
    grpo_records = fork_results["grpo"]["per_response_records"]
    dr_records = fork_results["dr_grpo"]["per_response_records"]
    shortest_grpo = min(grpo_records, key=lambda x: x["resp_len"])
    longest_grpo = max(grpo_records, key=lambda x: x["resp_len"])

    qual_records = [
        {
            "experimental_condition": "normalization_canonical_short",
            "prompt_id": shortest_grpo["prompt_id"],
            "prompt_text": f"prompt_{shortest_grpo['prompt_id']}",
            "group_id": shortest_grpo["update"],
            "K": 4,
            "completion": shortest_grpo["resp_text"],
            "reward": shortest_grpo["reward"],
            "completion_length": shortest_grpo["resp_len"],
            "relevant_statistic": f"policy_term_mag={shortest_grpo['policy_term_magnitude']:.4f}, per_token={shortest_grpo['per_token_contribution']:.6f}",
        },
        {
            "experimental_condition": "normalization_canonical_long",
            "prompt_id": longest_grpo["prompt_id"],
            "prompt_text": f"prompt_{longest_grpo['prompt_id']}",
            "group_id": longest_grpo["update"],
            "K": 4,
            "completion": longest_grpo["resp_text"],
            "reward": longest_grpo["reward"],
            "completion_length": longest_grpo["resp_len"],
            "relevant_statistic": f"policy_term_mag={longest_grpo['policy_term_magnitude']:.4f}, per_token={longest_grpo['per_token_contribution']:.6f}",
        },
    ]

    for rec in qual_records:
        append_jsonl(qual_file, rec)

    output = {
        "experimental_controls": {
            "seed": int(cfg["seed"]),
            "reward_model": cfg["reward_model"],
            "kl_beta": float(cfg["kl_beta"]),
            "clip_epsilon": float(cfg["clip_epsilon"]),
            "K": int(cfg.get("num_generations", 4)),
            "fork_updates": fork_updates,
            "max_completion_length": int(cfg.get("max_completion_length", 512)),
            "midpoint_checkpoint": cfg["paths"]["grpo_midpoint_policy"],
            "generation_settings": cfg.get("generation", {}),
        },
        "heldout_comparison": {
            "canonical_grpo": fork_results["grpo"]["heldout"],
            "dr_grpo": fork_results["dr_grpo"]["heldout"],
        },
        "length_conditioned_analysis": length_analysis,
        "update_trajectories": {
            "canonical_grpo": fork_results["grpo"]["updates"],
            "dr_grpo": fork_results["dr_grpo"]["updates"],
        },
        "per_response_records": {
            "canonical_grpo": grpo_records,
            "dr_grpo": dr_records,
        },
    }

    save_json(out_file, output)
    print(f"\nNormalization comparison complete. Results saved to: {out_file}")
    print(f"Qualitative examples appended to: {qual_file}")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_normalization_comparison(args.config)


if __name__ == "__main__":
    main()
