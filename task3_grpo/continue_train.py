from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
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


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_completions(policy, tokenizer, prompt_ids, K, max_new_tokens, gen_cfg, device):
    """Generate K completions for a single prompt. Returns list of token-id tensors."""
    inp = torch.tensor([prompt_ids], device=device)
    attn = torch.ones_like(inp)
    completions = []
    for _ in range(K):
        out = policy.generate(
            inp,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=gen_cfg.get("do_sample", True),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
        )
        completions.append(out[0][len(prompt_ids):].cpu())
    return completions


@torch.no_grad()
def score_completion(reward_model, reward_tokenizer, full_text, max_length, device):
    rm_in = reward_tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    return reward_model(**rm_in).logits[0, 0].item()


# ---------------------------------------------------------------------------
# Per-token log-prob computation
# ---------------------------------------------------------------------------

def get_logprobs(policy, input_ids, attention_mask, response_mask):
    """Return per-token log-probs for the response tokens only.

    Shape: [seq_len - 1]  (aligned to prediction targets)
    """
    logits = policy(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    labels = input_ids[:, 1:]               # [1, L-1]
    lp = logits[:, :-1, :].log_softmax(-1)  # [1, L-1, V]
    per_tok = torch.gather(lp, 2, labels.unsqueeze(2)).squeeze(2).squeeze(0)  # [L-1]
    mask = response_mask[:, 1:].squeeze(0)  # [L-1]
    return per_tok, mask


# ---------------------------------------------------------------------------
# Main GRPO loop
# ---------------------------------------------------------------------------

def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)

    out_dir = repo_path(output or cfg["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    res_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    res_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = res_dir / "training.jsonl"
    summary_path = res_dir / "training_summary.json"

    # Wipe old log for this run
    if jsonl_path.exists():
        jsonl_path.unlink()

    # ---- hyperparameters ------------------------------------------------
    total_updates       = int(cfg["updates"])
    K                   = int(cfg.get("num_generations", 4))
    max_prompt_len      = int(cfg.get("max_prompt_length", 256))
    max_comp_len        = int(cfg.get("max_completion_length", 256))
    clip_eps            = float(cfg["clip_epsilon"])
    kl_beta             = float(cfg["kl_beta"])
    max_grad_norm       = float(cfg.get("max_grad_norm", 1.0))
    mask_trunc          = bool(cfg.get("mask_truncated_completions", True))
    reward_max_len      = int(cfg.get("reward_max_length", 1280))
    gen_cfg             = cfg.get("generation", {})
    prompts_per_update  = int(cfg.get("prompts_per_update", 1))

    tokenizer       = bundle["tokenizer"]
    policy          = bundle["policy"]
    reward_model, r_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    prompts         = bundle["prompt_rows"]
    optimizer       = bundle["optimizer"]
    device          = next(policy.parameters()).device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timer = wall_timer()
    print(
        f"Running GRPO [{run_name}] | loss={loss_type} | updates={total_updates} "
        f"| K={K} | max_comp={max_comp_len} | eps={clip_eps} | kl_beta={kl_beta}"
    )

    history = []

    for upd in range(total_updates):
        t_upd = time.perf_counter()
        policy.eval()

        # ---- collect K completions for each prompt in this update --------
        all_rewards:    list[float]         = []
        all_group_ids:  list[int]           = []
        all_input_ids:  list[torch.Tensor]  = []
        all_attn_mask:  list[torch.Tensor]  = []
        all_resp_mask:  list[torch.Tensor]  = []
        all_truncated:  list[bool]          = []
        all_texts:      list[str]           = []
        all_resp_texts: list[str]           = []

        for pi in range(prompts_per_update):
            prompt_idx = (upd * prompts_per_update + pi) % len(prompts)
            row = prompts[prompt_idx]
            msgs = prompt_messages(row)
            prompt_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True
            )
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[:max_prompt_len]

            completions = generate_completions(
                policy, tokenizer, prompt_ids, K, max_comp_len, gen_cfg, device
            )

            for comp_ids in completions:
                # detect truncation: response fills max_comp_len and no EOS
                truncated = (
                    len(comp_ids) >= max_comp_len
                    and (tokenizer.eos_token_id is None
                         or comp_ids[-1].item() != tokenizer.eos_token_id)
                )
                all_truncated.append(truncated)

                # Build full sequence tensors
                full_ids = torch.tensor(
                    prompt_ids + comp_ids.tolist(), dtype=torch.long, device=device
                ).unsqueeze(0)
                attn = torch.ones_like(full_ids)
                resp_mask = torch.zeros_like(full_ids, dtype=torch.float32)
                resp_mask[0, len(prompt_ids):] = 1.0

                all_input_ids.append(full_ids)
                all_attn_mask.append(attn)
                all_resp_mask.append(resp_mask)
                all_group_ids.append(pi)

                # Reward
                full_text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
                resp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                all_texts.append(full_text)
                all_resp_texts.append(resp_text)

                rew = score_completion(
                    reward_model, r_tok, full_text, reward_max_len, reward_model.device
                )
                all_rewards.append(rew)

        # ---- per-group relative advantages -------------------------------
        rewards_t   = torch.tensor(all_rewards, dtype=torch.float32)
        group_ids_t = torch.tensor(all_group_ids, dtype=torch.long)
        seq_adv     = group_relative_advantages(rewards_t, group_ids_t)  # [N]

        # ---- group stats for logging ------------------------------------
        uninformative_count = 0
        group_reward_stds: list[float] = []
        for gid in group_ids_t.unique().tolist():
            gmask = group_ids_t == gid
            gr = rewards_t[gmask]
            g_std = gr.std(unbiased=False).item()
            group_reward_stds.append(g_std)
            if g_std < 1e-4:
                uninformative_count += 1
        n_groups = len(group_ids_t.unique())
        uninform_frac = uninformative_count / max(n_groups, 1)
        mean_group_reward_std = float(sum(group_reward_stds) / max(len(group_reward_stds), 1))

        # ---- compute old log-probs (no-grad) ----------------------------
        with torch.no_grad():
            old_logps: list[torch.Tensor] = []
            ref_logps: list[torch.Tensor] = []
            for i in range(len(all_input_ids)):
                olp, _  = get_logprobs(policy, all_input_ids[i], all_attn_mask[i], all_resp_mask[i])
                old_logps.append(olp.detach())
                with reference_mode(policy):
                    rlp, _ = get_logprobs(policy, all_input_ids[i], all_attn_mask[i], all_resp_mask[i])
                ref_logps.append(rlp.detach())

        # ---- PPO-style update epoch(s) -----------------------------------
        policy.train()
        policy_epochs = int(cfg.get("policy_epochs", 1))

        total_loss = 0.0
        total_p_term = 0.0
        total_kl = 0.0
        total_clip = 0.0
        total_ent = 0.0
        total_grad_norm = 0.0

        for _ep in range(policy_epochs):
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            batch_stats: dict[str, float] = {
                "policy_term": 0.0, "kl": 0.0, "clip": 0.0, "entropy": 0.0
            }

            for i in range(len(all_input_ids)):
                cur_logp, tok_mask = get_logprobs(
                    policy, all_input_ids[i], all_attn_mask[i], all_resp_mask[i]
                )

                # Expand masks to 2-D [1, L-1] as expected by grpo_policy_loss
                cur_logp_2d = cur_logp.unsqueeze(0)
                old_logp_2d = old_logps[i].unsqueeze(0)
                ref_logp_2d = ref_logps[i].unsqueeze(0)
                tok_mask_2d = tok_mask.unsqueeze(0)

                # Apply truncation masking if configured
                if mask_trunc and all_truncated[i]:
                    tok_mask_2d = mask_truncated_sequences(tok_mask_2d, [True])

                adv_i = seq_adv[i].unsqueeze(0).to(cur_logp_2d.device)

                loss_i, stats_i = grpo_policy_loss(
                    new_logp=cur_logp_2d,
                    old_logp=old_logp_2d,
                    seq_adv=adv_i,
                    token_mask=tok_mask_2d,
                    ref_logp=ref_logp_2d,
                    eps=clip_eps,
                    beta=kl_beta,
                    loss_type=loss_type,
                    max_completion_length=max_comp_len,
                )
                batch_loss = batch_loss + loss_i / len(all_input_ids)
                for k in batch_stats:
                    key_map = {"kl": "sampled_kl", "clip": "clip_fraction", "entropy": "sample_entropy", "policy_term": "policy_term"}
                    batch_stats[k] += stats_i[key_map[k]].item() / len(all_input_ids)

            if torch.isfinite(batch_loss):
                batch_loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm).item()
                optimizer.step()
            else:
                gn = float("nan")

            total_loss     += batch_loss.item()
            total_p_term   += batch_stats["policy_term"]
            total_kl       += batch_stats["kl"]
            total_clip     += batch_stats["clip"]
            total_ent      += batch_stats["entropy"]
            total_grad_norm+= gn

        div = max(policy_epochs, 1)
        mean_resp_len = float(
            sum(all_resp_mask[i].sum().item() for i in range(len(all_resp_mask)))
            / max(len(all_resp_mask), 1)
        )

        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        )

        record = {
            "update":                upd + 1,
            "run_name":              run_name,
            "loss_type":             loss_type,
            "mean_reward":           float(rewards_t.mean().item()),
            "std_reward":            float(rewards_t.std(unbiased=False).item()),
            "mean_group_reward_std": mean_group_reward_std,
            "uninformative_group_fraction": uninform_frac,
            "policy_loss":           total_loss / div,
            "policy_term":           total_p_term / div,
            "kl":                    total_kl / div,
            "clip_fraction":         total_clip / div,
            "entropy":               total_ent / div,
            "grad_norm":             total_grad_norm / div,
            "mean_response_length":  mean_resp_len,
            "peak_vram_gb":          peak_vram_gb,
            "elapsed_sec":           time.perf_counter() - t_upd,
        }
        history.append(record)
        append_jsonl(jsonl_path, record)

        print(
            f"Update {upd+1:>3}/{total_updates} | "
            f"Reward: {record['mean_reward']:+.4f} ± {record['std_reward']:.4f} | "
            f"KL: {record['kl']:.4f} | Loss: {record['policy_loss']:.6f} | "
            f"GradNorm: {record['grad_norm']:.4f} | "
            f"Uninform: {uninform_frac:.2f} | Len: {mean_resp_len:.1f}"
        )

    # ---- save adapter + summary -----------------------------------------
    policy.save_pretrained(str(out_dir))

    wall_time = timer()
    summary = {
        "run_name":            run_name,
        "loss_type":           loss_type,
        "total_updates":       len(history),
        "wall_clock_time_sec": wall_time,
        "peak_vram_gb":        peak_vram_gb,
        "config": {
            "updates":              total_updates,
            "K":                    K,
            "max_completion_length": max_comp_len,
            "max_prompt_length":    max_prompt_len,
            "clip_epsilon":         clip_eps,
            "kl_beta":              kl_beta,
            "learning_rate":        float(cfg["learning_rate"]),
            "max_grad_norm":        max_grad_norm,
            "mask_truncated_completions": mask_trunc,
            "seed":                 int(cfg["seed"]),
            "loss_type":            loss_type,
        },
        "history": history,
    }
    save_json(summary_path, summary)
    print(
        f"GRPO training done in {wall_time:.1f}s. "
        f"Adapter saved to {out_dir}. Summary: {summary_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates",   type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name",  default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
