from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.metrics import sampled_kl, sample_entropy, mean_response_length
from common.models import load_policy, load_reward_model, load_tokenizer


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "ref_policy": load_policy(cfg, adapter_path=None, trainable=False),
        "reward": load_reward_model(cfg),
    }


def get_per_token_logprobs(model, input_ids, attention_mask):
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return per_token_logps


def evaluate_ppo_policy(policy, ref_policy, reward_bundle, tokenizer, rows, cfg, run_name="standard"):
    reward_model, r_tokenizer = reward_bundle
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen = int(cfg.get("eval_max_response_length", 512))

    rewards = []
    lengths = []
    kls = []
    entropies = []
    examples = []

    for idx, row in enumerate(rows):
        prompt_msgs = prompt_messages(row)
        prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) > max_prompt:
            continue

        input_ids = torch.tensor([prompt_ids]).to(policy.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = policy.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        gen_ids = outputs[0][len(prompt_ids):]
        if len(gen_ids) == 0:
            continue

        resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        lengths.append(len(gen_ids))

        # Reward Model Score
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rm_inputs = r_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=int(cfg.get("reward_max_length", 1280))).to(reward_model.device)
        with torch.no_grad():
            score = reward_model(**rm_inputs).logits[0, 0].item()
        rewards.append(score)

        # Token-level KL and Entropy
        seq = outputs
        attn = torch.ones_like(seq)
        resp_mask = torch.zeros_like(seq)[:, 1:]
        resp_mask[0, len(prompt_ids) - 1 :] = 1

        with torch.no_grad():
            pol_logp = get_per_token_logprobs(policy, seq, attn)
            ref_logp = get_per_token_logprobs(ref_policy, seq, attn)

            kl_val = sampled_kl(pol_logp, ref_logp, resp_mask).item()
            ent_val = sample_entropy(pol_logp, resp_mask).item()
            kls.append(kl_val)
            entropies.append(ent_val)

        if len(examples) < 5:
            examples.append({
                "prompt": prompt_msgs[-1]["content"] if prompt_msgs else "",
                "response": resp_text,
                "reward": score,
                "kl": kl_val,
                "length": len(gen_ids),
            })

    metrics = {
        "run_name": run_name,
        "eval_count": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_response_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_response_length": float(np.std(lengths)) if lengths else 0.0,
        "qualitative_examples": examples,
    }

    res_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / f"eval_{run_name}.json"
    out_file.write_text(json.dumps(metrics, indent=2))
    print(f"Evaluation complete for [{run_name}]. Results saved to {out_file}")
    print(f"Mean Reward: {metrics['mean_reward']:.4f} | Mean KL: {metrics['mean_kl']:.4f} | Mean Length: {metrics['mean_response_length']:.2f}")

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()

    print(f"Loading evaluation bundle for adapter={args.adapter}...")
    bundle = load_evaluation_bundle(args.config, args.adapter)
    evaluate_ppo_policy(
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
