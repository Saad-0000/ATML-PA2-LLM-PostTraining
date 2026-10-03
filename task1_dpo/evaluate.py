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
    
    print("For full KL/Reward evaluation, generating completions is required. Implement generation and reward model scoring as needed by the assignment.")


if __name__ == "__main__":
    main()
