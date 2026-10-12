from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_yaml, repo_path


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    results_dir = repo_path(cfg["results_dir"])
    std_dir = results_dir / "standard"
    bal_dir = results_dir / "length_balanced"
    
    std_gen = {r["id"]: r for r in load_jsonl(std_dir / "generation_results.jsonl")}
    bal_gen = {r["id"]: r for r in load_jsonl(bal_dir / "generation_results.jsonl")}
    
    std_word = {r["id"]: r for r in load_jsonl(std_dir / "word_limit_results.jsonl")}
    bal_word = {r["id"]: r for r in load_jsonl(bal_dir / "word_limit_results.jsonl")}
    
    qualitative = []
    
    # 1. Differences between standard and length-balanced policies
    for eid in std_gen:
        if eid in bal_gen:
            r1 = std_gen[eid]
            r2 = bal_gen[eid]
            if abs(r1["length"] - r2["length"]) > 50:
                qualitative.append({
                    "phenomenon": "Length vs Standard Difference",
                    "id": eid,
                    "prompt": r1["prompt"],
                    "standard_response": r1["response"],
                    "standard_reward": r1["reward"],
                    "standard_length": r1["length"],
                    "balanced_response": r2["response"],
                    "balanced_reward": r2["reward"],
                    "balanced_length": r2["length"]
                })
                break
                
    # 2. Higher reward-model score does not correspond to better instruction following
    # Look for word limit compliance vs reward in word_limit_results (scored by reward model)
    for eid, word_r in std_word.items():
        if not word_r.get("compliant", True) and word_r.get("reward", float("-inf")) > 0:
            qualitative.append({
                "phenomenon": "High Reward but Failed Constraint",
                "id": eid,
                "prompt": word_r["prompt"],
                "response": word_r["response"],
                "reward": word_r.get("reward"),
                "length": word_r.get("length", word_r.get("word_count")),
                "word_count": word_r.get("word_count"),
                "compliant": word_r["compliant"],
                "limit": word_r["limit"],
            })
            break

    # If no positive reward failure, find any constraint failure in standard
    if not any(q["phenomenon"] == "High Reward but Failed Constraint" for q in qualitative):
        for eid, word_r in std_word.items():
            if not word_r.get("compliant", True):
                qualitative.append({
                    "phenomenon": "Constraint Failure on Word Limit",
                    "id": eid,
                    "prompt": word_r["prompt"],
                    "response": word_r["response"],
                    "reward": word_r.get("reward"),
                    "length": word_r.get("length", word_r.get("word_count")),
                    "word_count": word_r.get("word_count"),
                    "compliant": word_r["compliant"],
                    "limit": word_r["limit"],
                })
                break

    # 3. Word limit behavior: Standard vs Length-balanced comparison
    for eid in std_word:
        if eid in bal_word:
            w1 = std_word[eid]
            w2 = bal_word[eid]
            if w1.get("compliant") != w2.get("compliant") or abs(w1.get("word_count", 0) - w2.get("word_count", 0)) >= 5:
                qualitative.append({
                    "phenomenon": "Word Limit Confounding: Standard vs Length-Balanced",
                    "id": eid,
                    "prompt": w1["prompt"],
                    "limit": w1["limit"],
                    "standard_response": w1["response"],
                    "standard_words": w1.get("word_count"),
                    "standard_compliant": w1.get("compliant"),
                    "balanced_response": w2["response"],
                    "balanced_words": w2.get("word_count"),
                    "balanced_compliant": w2.get("compliant"),
                })
                break
                
    out_path = results_dir / "qualitative_examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qualitative, f, indent=2)
        
    print(f"Extracted {len(qualitative)} qualitative examples to {out_path}")


if __name__ == "__main__":
    main()

