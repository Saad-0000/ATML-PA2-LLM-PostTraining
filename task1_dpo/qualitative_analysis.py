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
    # Look for word limit compliance vs reward
    for eid in std_word:
        if eid in std_gen:
            word_r = std_word[eid]
            gen_r = std_gen[eid]
            if not word_r["compliant"] and gen_r["reward"] > 0: # High reward but failed constraint
                qualitative.append({
                    "phenomenon": "High Reward but Failed Constraint",
                    "id": eid,
                    "prompt": word_r["prompt"],
                    "response": word_r["response"],
                    "reward": gen_r["reward"],
                    "length": gen_r["length"],
                    "compliant": word_r["compliant"],
                    "limit": word_r["limit"]
                })
                break
                
    out_path = results_dir / "qualitative_examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qualitative, f, indent=2)
        
    print(f"Extracted {len(qualitative)} qualitative examples to {out_path}")


if __name__ == "__main__":
    main()

