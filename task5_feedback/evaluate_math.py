from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_tokenizer, clear_gpu
from common.generation import batch_generate
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def get_batches(rows, bsz):
    for i in range(0, len(rows), bsz):
        yield rows[i:i+bsz]


def generate_for_policy(cfg, rows, tokenizer, policy_name, bsz=8):
    model = load_frozen_policy(cfg, policy_name)
    
    results = []
    
    max_len = cfg.get("max_prompt_length", 512)
    max_new = cfg.get("max_new_tokens", 512)
    gen_kwargs = cfg.get("generation", {})
    do_sample = gen_kwargs.get("do_sample", False)
    
    # Force deterministic generation
    do_sample = False

    for batch in tqdm(get_batches(rows, bsz), desc=f"Generating {policy_name}"):
        prompts = [row["messages"] for row in batch]
        gen_out = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=max_len,
            max_new_tokens=max_new,
            do_sample=do_sample,
        )
        for i, row in enumerate(batch):
            resp = gen_out["responses"][i]
            res_len = gen_out["response_lengths"][i]
            
            extracted = extract_designated_final(resp)
            reward = exact_reward(resp, row["gold_final"])
            
            results.append({
                "problem_id": row.get("prompt_id", row.get("source_index", "")),
                "question": row["question"],
                "gold_answer": row["gold_final"],
                "policy": policy_name,
                "response": resp,
                "extracted_final": extracted,
                "parsing_status": "success" if extracted is not None else "failed",
                "verifier_reward": reward,
                "response_token_length": res_len,
            })
            
    clear_gpu(model)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    
    results_dir = Path("results/task5_feedback")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_responses = {}
    
    # Generate for SFT, RLVR, RLAIF
    for p in ["sft", "rlvr", "rlaif"]:
        out_file = results_dir / f"{args.dataset}_{p}_responses.jsonl"
        if out_file.exists():
            print(f"Loading cached {p} responses from {out_file}")
            with open(out_file, "r") as f:
                all_responses[p] = [json.loads(line) for line in f]
        else:
            res = generate_for_policy(cfg, rows, tokenizer, p, bsz=16)
            all_responses[p] = res
            with open(out_file, "w") as f:
                for r in res:
                    f.write(json.dumps(r) + "\n")
                    
    # Metrics computation
    metrics = {}
    for p in ["sft", "rlvr", "rlaif"]:
        res = all_responses[p]
        n_eval = len(res)
        n_parsed = sum(1 for r in res if r["parsing_status"] == "success")
        n_correct = sum(1 for r in res if r["verifier_reward"] > 0)
        mean_len = sum(r["response_token_length"] for r in res) / n_eval
        
        metrics[p] = {
            "exact_accuracy": n_correct / n_eval if n_eval > 0 else 0,
            "format_compliance": n_parsed / n_eval if n_eval > 0 else 0,
            "mean_length": mean_len,
            "n_eval": n_eval,
            "n_parsed": n_parsed,
            "n_correct": n_correct
        }
        
    print("\n--- Basic Metrics ---")
    for p, m in metrics.items():
        print(f"{p.upper()}: Acc={m['exact_accuracy']:.2%} Format={m['format_compliance']:.2%} Len={m['mean_length']:.1f}")

    # Pairwise AI judge evaluation (RLAIF vs SFT)
    judge = PairwiseAIJudge(cfg, cache_path=Path("results/task5_feedback/judge_cache.json"))
    
    rlaif_vs_sft_win = 0.0
    rlaif_vs_sft_tie = 0.0
    judge_verifier_agreement = 0
    total_comparisons = len(rows)
    
    pairwise_records = []
    
    for i in tqdm(range(total_comparisons), desc="Judging RLAIF vs SFT"):
        row = rows[i]
        problem_str = row["question"]
        
        rlaif_resp = all_responses["rlaif"][i]["response"]
        sft_resp = all_responses["sft"][i]["response"]
        
        pref = judge.compare(problem_str, rlaif_resp, sft_resp)
        # Pref is A (RLAIF), B (SFT), or TIE
        
        if pref == "A":
            rlaif_vs_sft_win += 1
        elif pref == "TIE":
            rlaif_vs_sft_tie += 1
            
        rlaif_reward = all_responses["rlaif"][i]["verifier_reward"]
        sft_reward = all_responses["sft"][i]["verifier_reward"]
        
        if rlaif_reward > sft_reward:
            verif_pref = "A"
        elif sft_reward > rlaif_reward:
            verif_pref = "B"
        else:
            verif_pref = "TIE"
            
        if pref == verif_pref:
            judge_verifier_agreement += 1
            
        pairwise_records.append({
            "problem_id": row.get("prompt_id", row.get("source_index", "")),
            "rlaif_reward": rlaif_reward,
            "sft_reward": sft_reward,
            "judge_pref": pref,
            "verifier_pref": verif_pref
        })

    win_rate = (rlaif_vs_sft_win + 0.5 * rlaif_vs_sft_tie) / total_comparisons
    agreement_rate = judge_verifier_agreement / total_comparisons
    
    metrics["rlaif_vs_sft_win_rate"] = win_rate
    metrics["verifier_judge_agreement"] = agreement_rate
    
    print("\n--- Pairwise Metrics ---")
    print(f"RLAIF vs SFT Win Rate: {win_rate:.2%}")
    print(f"Verifier-Judge Agreement: {agreement_rate:.2%}")
    
    with open(results_dir / f"{args.dataset}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(results_dir / f"{args.dataset}_pairwise_records.jsonl", "w") as f:
        for r in pairwise_records:
            f.write(json.dumps(r) + "\n")

    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
