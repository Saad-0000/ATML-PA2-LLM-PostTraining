from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.metrics import sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "ref_policy": load_policy(cfg, adapter_path=None, trainable=False),
        "reward": load_reward_model(cfg),
    }


def extract_input_ids(encoded) -> list[int]:
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, dict) and "input_ids" in encoded:
        val = encoded["input_ids"]
        if hasattr(val, "tolist"):
            val = val.tolist()
        if val and isinstance(val, list) and isinstance(val[0], list):
            val = val[0]
        return list(val)
    if hasattr(encoded, "tolist"):
        val = encoded.tolist()
        if val and isinstance(val, list) and isinstance(val[0], list):
            val = val[0]
        return list(val)
    if isinstance(encoded, list):
        if encoded and hasattr(encoded[0], "ids"):
            return list(encoded[0].ids)
        if encoded and isinstance(encoded[0], list):
            return list(encoded[0])
        return list(encoded)
    return list(encoded)


def evaluate_grpo_policy(
    policy,
    ref_policy,
    reward_bundle,
    tokenizer,
    rows,
    cfg,
    run_name: str = "standard",
):
    reward_model, r_tok = reward_bundle
    device = next(policy.parameters()).device
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen    = int(cfg.get("eval_max_response_length",
                             cfg.get("max_completion_length", 256)))
    reward_max = int(cfg.get("reward_max_length", 1280))

    rewards, lengths, kls, entropies = [], [], [], []
    examples = []

    for idx, row in enumerate(rows):
        msgs = prompt_messages(row)
        prompt_ids = extract_input_ids(tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        ))
        if len(prompt_ids) > max_prompt:
            continue

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn      = torch.ones_like(input_ids)

        with torch.no_grad():
            out = policy.generate(
                input_ids,
                attention_mask=attn,
                max_new_tokens=max_gen,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        gen_ids = out[0][len(prompt_ids):]
        if len(gen_ids) == 0:
            continue

        resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        lengths.append(len(gen_ids))

        # Reward
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        rm_in = r_tok(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=reward_max,
        ).to(reward_model.device)
        with torch.no_grad():
            score = reward_model(**rm_in).logits[0, 0].item()
        rewards.append(score)

        # KL and entropy
        seq  = out
        sattn = torch.ones_like(seq)
        resp_mask = torch.zeros_like(seq[:, 1:], dtype=torch.float32)
        resp_mask[0, len(prompt_ids) - 1:] = 1.0

        with torch.no_grad():
            def per_token_lp(model, s, sa):
                logits = model(input_ids=s, attention_mask=sa).logits.float()
                labels = s[:, 1:]
                return torch.gather(
                    logits[:, :-1, :].log_softmax(-1), 2, labels.unsqueeze(2)
                ).squeeze(2)

            pol_lp  = per_token_lp(policy, seq, sattn)
            with reference_mode(policy):
                ref_lp = per_token_lp(policy, seq, sattn)

        kl_val  = sampled_kl(pol_lp, ref_lp, resp_mask).item()
        ent_val = sample_entropy(pol_lp, resp_mask).item()
        kls.append(kl_val)
        entropies.append(ent_val)

        if len(examples) < 5:
            examples.append({
                "prompt_idx":  idx,
                "prompt":      msgs[-1]["content"] if msgs else "",
                "response":    resp_text,
                "reward":      score,
                "kl":          kl_val,
                "entropy":     ent_val,
                "length":      len(gen_ids),
            })

    metrics = {
        "run_name":            run_name,
        "eval_count":          len(rewards),
        "mean_reward":         float(np.mean(rewards)) if rewards else 0.0,
        "std_reward":          float(np.std(rewards))  if rewards else 0.0,
        "mean_kl":             float(np.mean(kls))     if kls else 0.0,
        "mean_entropy":        float(np.mean(entropies)) if entropies else 0.0,
        "mean_response_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_response_length": float(np.std(lengths))  if lengths else 0.0,
        "qualitative_examples": examples,
    }

    res_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / f"evaluation.json"
    save_json(out_file, metrics)
    print(
        f"Evaluation [{run_name}]: "
        f"Reward={metrics['mean_reward']:.4f} | "
        f"KL={metrics['mean_kl']:.4f} | "
        f"Len={metrics['mean_response_length']:.1f} | "
        f"N={metrics['eval_count']}"
    )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name",    default="standard")
    args = ap.parse_args()

    print(f"Loading evaluation bundle for adapter={args.adapter}...")
    bundle = load_evaluation_bundle(args.config, args.adapter)
    evaluate_grpo_policy(
        bundle["policy"],
        bundle["ref_policy"],
        bundle["reward"],
        bundle["tokenizer"],
        bundle["rows"],
        bundle["cfg"],
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
