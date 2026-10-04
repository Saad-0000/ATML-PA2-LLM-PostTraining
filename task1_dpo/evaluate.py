from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


import torch
from common.data import repo_path, encode_prompt_response, pad_batch, prompt_messages_from_preference, preference_responses, render_prompt
from common.models import reference_mode
import json

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

def evaluate_preference_accuracy(policy, tokenizer, rows, max_length):
    chosen_correct = 0
    total = 0
    
    for row in rows:
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
            with reference_mode(policy):
                ref_chosen_logp = get_logprobs(policy, c_batch)
                ref_rejected_logp = get_logprobs(policy, r_batch)
            
            pol_chosen_logp = get_logprobs(policy, c_batch)
            pol_rejected_logp = get_logprobs(policy, r_batch)
            
            policy_margin = pol_chosen_logp - pol_rejected_logp
            ref_margin = ref_chosen_logp - ref_rejected_logp
            
            if (policy_margin - ref_margin).item() > 0:
                chosen_correct += 1
            total += 1
            
    return chosen_correct / max(1, total) if total > 0 else 0

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

def evaluate_generation(policy, reward_bundle, tokenizer, rows, cfg):
    reward_model, r_tokenizer = reward_bundle
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen = int(cfg.get("max_generation_tokens", 256))
    
    rewards = []
    lengths = []
    kls = []
    
    for row in rows:
        prompt = prompt_messages_from_preference(row)
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) > max_prompt:
            continue
            
        input_ids = torch.tensor([prompt_ids]).to(policy.device)
        with torch.no_grad():
            outputs = policy.generate(
                input_ids, 
                max_new_tokens=max_gen, 
                pad_token_id=tokenizer.pad_token_id, 
                do_sample=False
            )
            
        gen_ids = outputs[0][len(prompt_ids):]
        if len(gen_ids) == 0:
            continue
            
        lengths.append(len(gen_ids))
        
        # Reward
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rm_inputs = r_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1280).to(reward_model.device)
        with torch.no_grad():
            score = reward_model(**rm_inputs).logits[0, 0].item()
        rewards.append(score)
        
        # KL
        seq = outputs
        attn = torch.ones_like(seq)
        resp_mask = torch.zeros_like(seq)[:, 1:] # mask for labels
        resp_mask[0, len(prompt_ids)-1:] = 1
        
        batch = {"input_ids": seq, "attention_mask": attn}
        with torch.no_grad():
            pol_logp = get_per_token_logprobs(policy, batch)
            with reference_mode(policy):
                ref_logp = get_per_token_logprobs(policy, batch)
                
            from common.metrics import sampled_kl
            kl = sampled_kl(pol_logp, ref_logp, resp_mask).item()
            kls.append(kl)
            
    import numpy as np
    print(f"Mean Reward: {np.mean(rewards):.4f}")
    print(f"Mean KL: {np.mean(kls):.4f}")
    print(f"Mean Length: {np.mean(lengths):.2f}")

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
    tokenizer = bundle["tokenizer"]
    rows = bundle["rows"]
    
    max_len = int(cfg["max_sequence_length"])
    
    print("Evaluating preference accuracy...")
    pref_acc = evaluate_preference_accuracy(policy, tokenizer, rows, max_len)
    print(f"[{args.name}] Preference Accuracy: {pref_acc:.4f}")
    
    print("Generating completions for KL and Reward scoring...")
    evaluate_generation(policy, bundle["reward"], tokenizer, rows, cfg)

if __name__ == "__main__":
    main()
