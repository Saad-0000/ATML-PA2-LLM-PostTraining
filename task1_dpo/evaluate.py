from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from common.data import load_yaml, read_jsonl, repo_path, encode_prompt_response, pad_batch, prompt_messages_from_preference, preference_responses
from common.models import load_policy, load_reward_model, load_tokenizer
from common.metrics import sampled_kl
from task1_dpo.dpo import dpo_loss


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "word_limit_rows": read_jsonl(cfg["paths"]["word_limit_prompts"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "ref_policy": load_policy(cfg, adapter_path=None, trainable=False),
        "reward": load_reward_model(cfg),
    }


def get_logprobs(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        response_mask = response_mask.cuda()
        
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    loss_mask = response_mask[:, 1:]
    
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)


def get_per_token_logprobs(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return per_token_logps


def count_words(text):
    return len(text.strip().split())


def extract_word_limit(prompt_text):
    import re
    m = re.search(r'in (\d+) words or less', prompt_text.lower())
    if m:
        return int(m.group(1))
    return None


def evaluate_preferences(policy, ref_policy, tokenizer, rows, cfg):
    max_length = int(cfg["max_sequence_length"])
    beta = float(cfg["beta"])
    
    results = []
    
    for row in tqdm(rows, desc="Evaluating preferences"):
        prompt = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)
        
        try:
            c_enc = encode_prompt_response(tokenizer, prompt, yc, max_length)
            r_enc = encode_prompt_response(tokenizer, prompt, yr, max_length)
        except ValueError:
            continue
            
        c_batch = pad_batch(tokenizer, [c_enc])
        r_batch = pad_batch(tokenizer, [r_enc])
        
        with torch.no_grad():
            ref_chosen_logp = get_logprobs(ref_policy, c_batch)
            ref_rejected_logp = get_logprobs(ref_policy, r_batch)
            
            pol_chosen_logp = get_logprobs(policy, c_batch)
            pol_rejected_logp = get_logprobs(policy, r_batch)
            
            loss, metrics = dpo_loss(
                pol_chosen_logp, pol_rejected_logp,
                ref_chosen_logp, ref_rejected_logp,
                beta
            )
            
        is_correct = metrics["preference_accuracy"].item() > 0
        
        results.append({
            "id": row.get("id", ""),
            "category": row.get("category", "unlabeled"),
            "loss": loss.item(),
            "preference_correct": is_correct,
            "policy_margin": metrics["policy_margin_mean"].item(),
            "chosen_length": count_words(yc),
            "rejected_length": count_words(yr)
        })
        
    return results


def evaluate_generation(policy, ref_policy, reward_bundle, tokenizer, rows, cfg):
    reward_model, r_tokenizer = reward_bundle
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen = int(cfg.get("max_generation_tokens", 256))
    
    results = []
    
    for row in tqdm(rows, desc="Evaluating generation"):
        prompt = prompt_messages_from_preference(row)
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
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
            )
            
        gen_ids = outputs[0][len(prompt_ids):]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        response_len = len(gen_ids)
        
        # Reward
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rm_inputs = r_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1280).to(reward_model.device)
        with torch.no_grad():
            score = reward_model(**rm_inputs).logits[0, 0].item()
            
        # KL calculation
        seq = outputs
        attn = torch.ones_like(seq)
        resp_mask = torch.zeros_like(seq)[:, 1:]
        resp_mask[0, len(prompt_ids)-1:] = 1
        
        batch = {"input_ids": seq, "attention_mask": attn}
        with torch.no_grad():
            ref_logp = get_per_token_logprobs(ref_policy, batch)
            pol_logp = get_per_token_logprobs(policy, batch)
            kl = sampled_kl(pol_logp, ref_logp, resp_mask).item()
            
        results.append({
            "id": row.get("id", ""),
            "category": row.get("category", "unlabeled"),
            "prompt": prompt,
            "response": response_text,
            "reward": score,
            "kl": kl,
            "length": response_len,
            "word_count": count_words(response_text)
        })
        
    return results


def evaluate_word_limits(policy, tokenizer, rows, cfg):
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen = int(cfg.get("max_generation_tokens", 256))
    
    results = []
    
    for row in tqdm(rows, desc="Evaluating word limits"):
        if isinstance(row.get("prompt"), list):
            prompt = row["prompt"]
            prompt_text = prompt[-1]["content"] if prompt else ""
        else:
            prompt_text = str(row.get("prompt", row.get("question", "")))
            prompt = [{"role": "user", "content": prompt_text}]
            
        limit = extract_word_limit(prompt_text)
        if not limit:
            continue
            
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
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
            )
            
        gen_ids = outputs[0][len(prompt_ids):]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        words = count_words(response_text)
        
        results.append({
            "id": row.get("id", ""),
            "prompt": prompt_text,
            "response": response_text,
            "limit": limit,
            "words": words,
            "compliant": words <= limit
        })
        
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    
    print(f"Loading models for {args.name} evaluation...")
    bundle = load_evaluation_bundle(args.config, args.adapter)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    ref_policy = bundle["ref_policy"]
    tokenizer = bundle["tokenizer"]
    
    output_dir = repo_path(cfg["results_dir"]) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Evaluating preference accuracy and DPO loss...")
    pref_results = evaluate_preferences(policy, ref_policy, tokenizer, bundle["rows"], cfg)
    with open(output_dir / "preference_results.jsonl", "w") as f:
        for r in pref_results:
            f.write(json.dumps(r) + "\n")
            
    print("Evaluating generation (Reward, KL, Length)...")
    gen_results = evaluate_generation(policy, ref_policy, bundle["reward"], tokenizer, bundle["rows"], cfg)
    with open(output_dir / "generation_results.jsonl", "w") as f:
        for r in gen_results:
            f.write(json.dumps(r) + "\n")
            
    print("Evaluating word limits...")
    word_results = evaluate_word_limits(policy, tokenizer, bundle["word_limit_rows"], cfg)
    with open(output_dir / "word_limit_results.jsonl", "w") as f:
        for r in word_results:
            f.write(json.dumps(r) + "\n")
            
    # Aggregate metrics
    summary = {
        "run_name": args.name,
        "n_pref_eval": len(pref_results),
        "n_gen_eval": len(gen_results),
        "n_word_eval": len(word_results)
    }
    
    if pref_results:
        summary["dpo_loss"] = np.mean([r["loss"] for r in pref_results])
        summary["preference_accuracy"] = np.mean([r["preference_correct"] for r in pref_results])
        
        # Stratified
        cats = defaultdict(list)
        for r in pref_results:
            cats[r["category"]].append(r)
        summary["stratified_pref_acc"] = {
            cat: np.mean([r["preference_correct"] for r in cat_results])
            for cat, cat_results in cats.items()
        }
        
    if gen_results:
        summary["reward_mean"] = np.mean([r["reward"] for r in gen_results])
        summary["kl_mean"] = np.mean([r["kl"] for r in gen_results])
        summary["length_mean"] = np.mean([r["length"] for r in gen_results])
        summary["length_std"] = np.std([r["length"] for r in gen_results])
        
        cats = defaultdict(list)
        for r in gen_results:
            cats[r["category"]].append(r)
        summary["stratified_reward"] = {
            cat: np.mean([r["reward"] for r in cat_results])
            for cat, cat_results in cats.items()
        }
        summary["stratified_length"] = {
            cat: np.mean([r["length"] for r in cat_results])
            for cat, cat_results in cats.items()
        }
        
    if word_results:
        summary["word_limit_compliance"] = np.mean([r["compliant"] for r in word_results])
        
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
