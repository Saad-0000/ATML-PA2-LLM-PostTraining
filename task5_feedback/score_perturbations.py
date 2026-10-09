from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from common.data import load_yaml, read_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def get_verifier_preference(a_reward, b_reward):
    if a_reward > b_reward:
        return "A"
    elif b_reward > a_reward:
        return "B"
    return "TIE"


def get_win_rate(pref, target="A"):
    if pref == target:
        return 1.0
    if pref == "TIE":
        return 0.5
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    
    judge = PairwiseAIJudge(cfg, cache_path=Path("results/task5_feedback/diagnostic_judge_cache.json"))
    
    all_pairs_records = []
    
    total_comparisons = 0
    ties_count = 0
    wrong_preference_count = 0
    wrong_preference_possible = 0
    
    # Track specific comparisons across all problems
    comparisons = {
        "s_reason": ("clean_correct", "corrupt_reasoning_correct_final"),
        "s_outcome": ("clean_correct", "good_reasoning_wrong_final"),
        "filler": ("clean_correct", "persuasive_filler_correct"),
        "distractor": ("good_reasoning_wrong_final", "gold_distractor_wrong_final")
    }
    
    metric_sums = defaultdict(lambda: {"judge_pref_A": 0.0, "verifier_pref_A": 0.0, "count": 0})
    
    for pid, variants in tqdm(groups.items(), desc="Evaluating diagnostics"):
        problem_str = variants["clean_correct"]["question"]
        gold = str(variants["clean_correct"]["gold_final"])
        
        variant_names = list(EXPECTED_VARIANTS)
        
        # Evaluate all pairs for tie and wrong preference rates
        for vA, vB in itertools.combinations(variant_names, 2):
            respA = variants[vA]["response"]
            respB = variants[vB]["response"]
            
            judge_pref = judge.compare(problem_str, respA, respB)
            total_comparisons += 1
            if judge_pref == "TIE":
                ties_count += 1
                
            rewardA = exact_reward(respA, gold)
            rewardB = exact_reward(respB, gold)
            
            if rewardA != rewardB:
                wrong_preference_possible += 1
                if rewardA > rewardB and judge_pref == "B":
                    wrong_preference_count += 1
                elif rewardB > rewardA and judge_pref == "A":
                    wrong_preference_count += 1
                    
            all_pairs_records.append({
                "problem_id": pid,
                "variantA": vA,
                "variantB": vB,
                "rewardA": rewardA,
                "rewardB": rewardB,
                "judge_pref": judge_pref
            })
            
        # Specific comparisons
        for metric_name, (vA, vB) in comparisons.items():
            respA = variants[vA]["response"]
            respB = variants[vB]["response"]
            
            judge_pref = judge.compare(problem_str, respA, respB)
            rewardA = exact_reward(respA, gold)
            rewardB = exact_reward(respB, gold)
            verif_pref = get_verifier_preference(rewardA, rewardB)
            
            judge_score_A = get_win_rate(judge_pref, "A")
            verif_score_A = get_win_rate(verif_pref, "A")
            
            metric_sums[metric_name]["judge_pref_A"] += judge_score_A
            metric_sums[metric_name]["verifier_pref_A"] += verif_score_A
            metric_sums[metric_name]["count"] += 1

    results_dir = Path("results/task5_feedback")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "tie_rate": ties_count / total_comparisons if total_comparisons > 0 else 0,
        "wrong_preference_rate": wrong_preference_count / wrong_preference_possible if wrong_preference_possible > 0 else 0
    }
    
    print(f"\nTie Rate: {metrics['tie_rate']:.1%}")
    print(f"Wrong Preference Rate: {metrics['wrong_preference_rate']:.1%} (Out of {wrong_preference_possible} where one is objectively correct)")
    
    for metric_name, data in metric_sums.items():
        n = data["count"]
        j_pref = data["judge_pref_A"] / n
        v_pref = data["verifier_pref_A"] / n
        diff = j_pref - v_pref
        metrics[f"{metric_name}_judge_pref"] = j_pref
        metrics[f"{metric_name}_verif_pref"] = v_pref
        metrics[f"{metric_name}_sensitivity"] = diff
        
        print(f"\n--- {metric_name} ({comparisons[metric_name][0]} vs {comparisons[metric_name][1]}) ---")
        print(f"Judge Pref for A:    {j_pref:.1%}")
        print(f"Verifier Pref for A: {v_pref:.1%}")
        print(f"Sensitivity (Diff):  {diff:+.1%}")
        
    with open(results_dir / "diagnostic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    with open(results_dir / "diagnostic_all_pairs.jsonl", "w") as f:
        for r in all_pairs_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nDiagnostic results saved to {results_dir}")


if __name__ == "__main__":
    main()
