from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.metrics import sample_entropy, masked_mean, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    reference_mode,
    token_values,
)
from task2_ppo.ppo import (
    compute_gae,
    shaped_rewards,
    ppo_policy_loss,
    value_mse_loss,
    normalize_advantages,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    # Cast the entire value model to float32 to prevent AdamW overflow.
    # The score head gradient can reach ~462 (max), whose square (213,444) far
    # exceeds float16's max representable value (65,504). AdamW stores the
    # second moment (exp_avg_sq) in float32 internally, but the final update
    # `param -= lr * update` is cast back to float16 if param.dtype is float16,
    # which overflows to Inf and corrupts all subsequent forward passes.
    # Keeping the whole value model in float32 keeps both parameters AND hidden
    # states in the same dtype, so no dtype-mismatch errors occur either.
    value_model = value_model.float()

    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def stats(tensor, mask=None):
    if mask is not None:
        t = tensor[mask.bool()].detach().cpu().float()
    else:
        t = tensor.detach().cpu().float()
    if t.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", verbose_diagnostics: bool = True):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running PPO continuation [{run_name}] with updates={cfg['updates']}, clip_epsilon={cfg['clip_epsilon']}, kl_beta={cfg['kl_beta']}")
    
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    value_coef = float(cfg.get("value_coef", 0.50))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_resp_len = int(cfg.get("max_response_length", 512))
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))
    
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model, r_tokenizer = bundle["reward_model"], bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]
    
    trainable_policy_params = trainable_parameters(policy)
    trainable_value_params = trainable_parameters(value_model)

    # Sanity checks
    assert len(trainable_policy_params) > 0, "Policy must have trainable parameters!"
    assert len(trainable_value_params) > 0, "Value model must have trainable parameters!"

    history = []
    
    for update_idx in range(int(cfg["updates"])):
        policy.eval()
        value_model.eval()
        
        # 1. Rollout Collection
        row = prompts[update_idx % len(prompts)]
        prompt_msgs = prompt_messages(row)
        prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]
            
        inp_tensor = torch.tensor([prompt_ids]).to(policy.device)
        attn_tensor = torch.ones_like(inp_tensor)
        
        with torch.no_grad():
            gen_out = policy.generate(
                inp_tensor,
                attention_mask=attn_tensor,
                max_new_tokens=max_resp_len,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
        resp_ids = gen_out[0][len(prompt_ids):]
        if len(resp_ids) == 0:
            continue
            
        seq = gen_out
        seq_attn = torch.ones_like(seq)
        resp_mask = torch.zeros_like(seq, dtype=torch.float32)
        resp_mask[0, len(prompt_ids):] = 1.0
        loss_mask = resp_mask[:, 1:] # [1, seq_len-1]
        
        # EOS Penalty check
        task_eos_penalty = 0.0
        if tokenizer.eos_token_id is not None and resp_ids[-1].item() != tokenizer.eos_token_id:
            task_eos_penalty = missing_eos_penalty

        # 2. Reward Scoring
        full_text = tokenizer.decode(seq[0], skip_special_tokens=True)
        rm_in = r_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=int(cfg.get("reward_max_length", 1280))).to(reward_model.device)
        with torch.no_grad():
            task_rew = reward_model(**rm_in).logits[0, 0].item() - task_eos_penalty
            
        # 3. Logprobs & Values computation
        with torch.no_grad():
            logits = policy(input_ids=seq, attention_mask=seq_attn).logits.float()
            labels = seq[:, 1:]
            log_probs = logits[:, :-1, :].log_softmax(-1)
            old_logp = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            
            with reference_mode(policy):
                ref_logits = policy(input_ids=seq, attention_mask=seq_attn).logits.float()
                ref_log_probs = ref_logits[:, :-1, :].log_softmax(-1)
                ref_logp = torch.gather(ref_log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
                
            all_vals = token_values(value_model, seq, seq_attn).float()
            old_values = all_vals[:, :-1]
            
        # Sanity assertions
        assert torch.isfinite(old_logp).all(), "old_logp contains non-finite values!"
        assert torch.isfinite(ref_logp).all(), "ref_logp contains non-finite values!"
        assert torch.isfinite(old_values).all(), "old_values contains non-finite values!"

        # 4. GAE & Shaped Rewards
        t_rew_tensor = torch.tensor([task_rew], device=policy.device, dtype=torch.float32)
        rewards = shaped_rewards(t_rew_tensor, old_logp, ref_logp, loss_mask, float(cfg["kl_beta"]))
        raw_advantages, returns = compute_gae(rewards, old_values, loss_mask, gamma, lam)
        advantages = normalize_advantages(raw_advantages, loss_mask)
        
        assert torch.isfinite(advantages).all(), "advantages contains non-finite values!"
        assert torch.isfinite(returns).all(), "returns contains non-finite values!"

        if update_idx == 0 and verbose_diagnostics:
            print("\n=== UPDATE 1 DETAILED DIAGNOSTIC INSTRUMENTATION ===")
            print(f"Valid Response Tokens: {int(loss_mask.sum().item())}")
            print(f"Nonzero Rewards Count: {int((rewards * loss_mask != 0).sum().item())}")
            print(f"Nonzero Advantages Count: {int((advantages * loss_mask != 0).sum().item())}")
            print("old_logp stats:", stats(old_logp, loss_mask))
            print("ref_logp stats:", stats(ref_logp, loss_mask))
            print("old_values stats:", stats(old_values, loss_mask))
            print("rewards stats:", stats(rewards, loss_mask))
            print("adv BEFORE norm:", stats(raw_advantages, loss_mask))
            print("adv AFTER norm:", stats(advantages, loss_mask))
            print("returns stats:", stats(returns, loss_mask))
            print("====================================================\n")

        # 5. PPO Update Epochs
        policy.train()
        value_model.train()
        
        pol_loss_val = 0.0
        val_loss_val = 0.0
        clip_frac_val = 0.0
        grad_norm_val = 0.0
        
        for epoch_idx in range(ppo_epochs):
            # Capture parameters before step in float32 for high precision comparison
            pol_params_before = [p.float().clone().detach() for p in trainable_policy_params]
            val_params_before = [p.float().clone().detach() for p in trainable_value_params]

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            
            cur_logits = policy(input_ids=seq, attention_mask=seq_attn).logits.float()
            cur_logp = torch.gather(cur_logits[:, :-1, :].log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            cur_vals = token_values(value_model, seq, seq_attn)[:, :-1].float()
            
            p_loss, ratio, c_frac = ppo_policy_loss(cur_logp, old_logp, advantages, loss_mask, float(cfg["clip_epsilon"]))
            v_loss = value_mse_loss(cur_vals, returns, loss_mask)
            
            tot_loss = p_loss + value_coef * v_loss
            if torch.isfinite(tot_loss):
                tot_loss.backward()
                
                p_norm = torch.nn.utils.clip_grad_norm_(trainable_policy_params, max_grad_norm).item()
                v_norm = torch.nn.utils.clip_grad_norm_(trainable_value_params, max_grad_norm).item()
                
                # Count nonzero gradients
                p_nonzero_grads = sum((p.grad != 0).sum().item() for p in trainable_policy_params if p.grad is not None)
                p_total_params = sum(p.numel() for p in trainable_policy_params)
                v_nonzero_grads = sum((p.grad != 0).sum().item() for p in trainable_value_params if p.grad is not None)
                v_total_params = sum(p.numel() for p in trainable_value_params)

                policy_optimizer.step()
                value_optimizer.step()
                
                # Parameter change computed in float32
                pol_l2_change = torch.sqrt(sum(((p.float() - pb) ** 2).sum() for p, pb in zip(trainable_policy_params, pol_params_before))).item()
                val_l2_change = torch.sqrt(sum(((v.float() - vb) ** 2).sum() for v, vb in zip(trainable_value_params, val_params_before))).item()

                if update_idx == 0 and verbose_diagnostics:
                    ratio_mask = ratio[loss_mask.bool()]
                    eps = float(cfg["clip_epsilon"])
                    below_clip = (ratio_mask < (1.0 - eps)).float().mean().item()
                    above_clip = (ratio_mask > (1.0 + eps)).float().mean().item()
                    print(f"PPO Epoch {epoch_idx + 1} | Loss: {p_loss.item():.6f} | Ratio Stats: {stats(ratio, loss_mask)} | Clip Frac: {c_frac.item():.4f} (Below: {below_clip:.4f}, Above: {above_clip:.4f})")
                    print(f"PPO Epoch {epoch_idx + 1} | Policy Grad Norm: {p_norm:.6f} | Nonzero Grads: {p_nonzero_grads}/{p_total_params} | Pol Param L2 Change: {pol_l2_change:.6e}")
                    print(f"PPO Epoch {epoch_idx + 1} | Value Grad Norm: {v_norm:.6f} | Nonzero Grads: {v_nonzero_grads}/{v_total_params} | Val Param L2 Change: {val_l2_change:.6e}")

                pol_loss_val += p_loss.item()
                val_loss_val += v_loss.item()
                clip_frac_val += c_frac.item()
                grad_norm_val += (p_norm + v_norm) / 2.0
            
        pol_loss_val /= ppo_epochs
        val_loss_val /= ppo_epochs
        clip_frac_val /= ppo_epochs
        grad_norm_val /= ppo_epochs
        
        mean_kl = sampled_kl(old_logp, ref_logp, loss_mask).item()
        mean_ent = sample_entropy(old_logp, loss_mask).item()
        resp_len = float(loss_mask.sum().item())
        
        update_metrics = {
            "update": update_idx + 1,
            "reward": task_rew,
            "kl": mean_kl,
            "entropy": mean_ent,
            "policy_loss": pol_loss_val,
            "value_loss": val_loss_val,
            "clip_fraction": clip_frac_val,
            "grad_norm": grad_norm_val,
            "response_length": resp_len,
        }
        history.append(update_metrics)
        print(f"Update {update_idx+1}/{cfg['updates']} | Task Reward: {task_rew:.4f} | KL: {mean_kl:.4f} | Policy Loss: {pol_loss_val:.6f} | Value Loss: {val_loss_val:.6f} | Clip Frac: {clip_frac_val:.4f}")
        
    wall_time = time.time() - start_time
    peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_vram_gb = peak_vram_bytes / (1024 ** 3)
    
    summary = {
        "run_name": run_name,
        "total_updates": len(history),
        "wall_clock_time_sec": wall_time,
        "peak_vram_gb": peak_vram_gb,
        "history": history,
    }
    
    policy.save_pretrained(out)
    
    res_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = res_dir / f"history_{run_name}.json"
    log_path.write_text(json.dumps(summary, indent=2))
    print(f"PPO training finished in {wall_time:.2f}s. Peak VRAM: {peak_vram_gb:.2f} GB. Saved to {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
