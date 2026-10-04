from __future__ import annotations

import argparse
from pathlib import Path
import torch

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
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


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
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
    
    import time
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    from task2_ppo.ppo import (
        compute_gae,
        shaped_rewards,
        ppo_policy_loss,
        value_mse_loss,
        normalize_advantages,
    )
    from common.models import reference_mode, token_values
    from common.metrics import sample_entropy, masked_mean, sampled_kl
    import numpy as np
    import json

    gamma = float(cfg.get("gamma", 1.0))
    lam = float(cfg.get("gae_lambda", 0.95))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    value_coef = float(cfg.get("value_coef", 0.50))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_resp_len = int(cfg.get("max_response_length", 512))
    
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model, r_tokenizer = bundle["reward_model"], bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]
    
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
        
        # 2. Reward Scoring
        full_text = tokenizer.decode(seq[0], skip_special_tokens=True)
        rm_in = r_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=int(cfg.get("reward_max_length", 1280))).to(reward_model.device)
        with torch.no_grad():
            task_rew = reward_model(**rm_in).logits[0, 0].item()
            
        # 3. Logprobs & Values computation
        with torch.no_grad():
            # Policy old logprobs
            logits = policy(input_ids=seq, attention_mask=seq_attn).logits
            labels = seq[:, 1:]
            log_probs = logits[:, :-1, :].log_softmax(-1)
            old_logp = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            
            # Reference logprobs
            with reference_mode(policy):
                ref_logits = policy(input_ids=seq, attention_mask=seq_attn).logits
                ref_log_probs = ref_logits[:, :-1, :].log_softmax(-1)
                ref_logp = torch.gather(ref_log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
                
            # Values
            all_vals = token_values(value_model, seq, seq_attn) # [1, seq_len]
            old_values = all_vals[:, :-1]
            
        # Slice for response positions
        loss_mask = resp_mask[:, 1:] # [1, seq_len-1]
        
        # 4. GAE & Shaped Rewards
        t_rew_tensor = torch.tensor([task_rew], device=policy.device, dtype=torch.float32)
        rewards = shaped_rewards(t_rew_tensor, old_logp, ref_logp, loss_mask, float(cfg["kl_beta"]))
        advantages, returns = compute_gae(rewards, old_values, loss_mask, gamma, lam)
        advantages = normalize_advantages(advantages, loss_mask)
        
        # 5. PPO Update Epochs
        policy.train()
        value_model.train()
        
        pol_loss_val = 0.0
        val_loss_val = 0.0
        clip_frac_val = 0.0
        grad_norm_val = 0.0
        
        for _ in range(ppo_epochs):
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            
            # Forward pass
            cur_logits = policy(input_ids=seq, attention_mask=seq_attn).logits
            cur_logp = torch.gather(cur_logits[:, :-1, :].log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            
            cur_vals = token_values(value_model, seq, seq_attn)[:, :-1]
            
            p_loss, ratio, c_frac = ppo_policy_loss(cur_logp, old_logp, advantages, loss_mask, float(cfg["clip_epsilon"]))
            v_loss = value_mse_loss(cur_vals, returns, loss_mask)
            
            tot_loss = p_loss + value_coef * v_loss
            tot_loss.backward()
            
            p_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm).item()
            v_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm).item()
            
            policy_optimizer.step()
            value_optimizer.step()
            
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
        print(f"Update {update_idx+1}/{cfg['updates']} | Task Reward: {task_rew:.4f} | KL: {mean_kl:.4f} | Policy Loss: {pol_loss_val:.4f} | Value Loss: {val_loss_val:.4f} | Clip Frac: {clip_frac_val:.4f}")
        
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
