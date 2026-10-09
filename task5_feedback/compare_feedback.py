from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_yaml


def extract_qualitative_examples(results_dir: Path):
    examples = []
    
    # 1. RLAIF hallucinating correctness (judge prefers incorrect RLAIF over correct SFT)
    # This requires looking at GSM8K pairwise records and responses.
    try:
        gsm_records = [json.loads(line) for line in open(results_dir / "gsm_pairwise_records.jsonl")]
        sft_responses = [json.loads(line) for line in open(results_dir / "gsm_sft_responses.jsonl")]
        rlaif_responses = [json.loads(line) for line in open(results_dir / "gsm_rlaif_responses.jsonl")]
        
        # Build dictionaries for easy access
        sft_dict = {r["problem_id"]: r for r in sft_responses}
        rlaif_dict = {r["problem_id"]: r for r in rlaif_responses}
        
        phenom1 = None
        for record in gsm_records:
            pid = record["problem_id"]
            # SFT is correct, RLAIF is incorrect, but judge prefers RLAIF ("A")
            # Wait, in evaluate_math, A is RLAIF, B is SFT.
            if record["sft_reward"] == 1 and record["rlaif_reward"] == 0 and record["judge_pref"] == "A":
                phenom1 = {
                    "phenomenon": "RLAIF hallucinating correctness",
                    "problem_id": pid,
                    "prompt": sft_dict[pid]["question"],
                    "sft_response": sft_dict[pid]["response"],
                    "rlaif_response": rlaif_dict[pid]["response"],
                    "sft_reward": record["sft_reward"],
                    "rlaif_reward": record["rlaif_reward"],
                    "judge_pref": record["judge_pref"]
                }
                break
        if phenom1:
            examples.append(phenom1)
    except Exception as e:
        print(f"Skipping Phenomenon 1 extraction due to missing data: {e}")

    # 2. RLAIF penalizing correct reasoning (Judge prefers corrupt over clean)
    try:
        diag_records = [json.loads(line) for line in open(results_dir / "diagnostic_all_pairs.jsonl")]
        
        phenom2 = None
        for record in diag_records:
            # We want variantA=clean_correct, variantB=corrupt_reasoning_correct_final or vice versa
            vA = record["variantA"]
            vB = record["variantB"]
            if {vA, vB} == {"clean_correct", "corrupt_reasoning_correct_final"}:
                # If judge prefers corrupt over clean
                pref_corrupt = (vA == "corrupt_reasoning_correct_final" and record["judge_pref"] == "A") or \
                               (vB == "corrupt_reasoning_correct_final" and record["judge_pref"] == "B")
                if pref_corrupt:
                    phenom2 = {
                        "phenomenon": "RLAIF penalizing correct reasoning",
                        "problem_id": record["problem_id"],
                        "variantA": vA,
                        "variantB": vB,
                        "judge_pref": record["judge_pref"]
                    }
                    break
        if phenom2:
            examples.append(phenom2)
    except Exception as e:
        print(f"Skipping Phenomenon 2 extraction due to missing data: {e}")
        
    # 3. Reward hacking in RLVR (Correct final answer, broken reasoning)
    # Find any RLVR response that got reward=1.
    try:
        rlvr_responses = [json.loads(line) for line in open(results_dir / "gsm_rlvr_responses.jsonl")]
        phenom3 = None
        for r in rlvr_responses:
            if r["verifier_reward"] == 1:
                phenom3 = {
                    "phenomenon": "Reward hacking in RLVR",
                    "problem_id": r["problem_id"],
                    "prompt": r["question"],
                    "rlvr_response": r["response"],
                    "verifier_reward": r["verifier_reward"],
                    "note": "Manually inspect if this contains reward hacking. If not, pick another."
                }
                break
        if phenom3:
            examples.append(phenom3)
    except Exception as e:
        print(f"Skipping Phenomenon 3 extraction due to missing data: {e}")

    with open(results_dir / "qualitative_examples.json", "w") as f:
        json.dump(examples, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    
    cfg = load_yaml(args.config)
    results_dir = Path("results/task5_feedback")
    
    # 1. Qualitative Extraction
    extract_qualitative_examples(results_dir)
    
    # 2. Print Summary Report
    print("=== TASK 5: FEEDBACK SOURCE SUMMARY ===")
    try:
        gsm_metrics = json.loads(open(results_dir / "gsm_metrics.json").read())
        print("\n--- GSM8K In-Domain ---")
        for p in ["sft", "rlvr", "rlaif"]:
            if p in gsm_metrics:
                print(f"{p.upper()}: Acc={gsm_metrics[p]['exact_accuracy']:.2%} Format={gsm_metrics[p]['format_compliance']:.2%} Len={gsm_metrics[p]['mean_length']:.1f}")
        print(f"RLAIF vs SFT Win Rate: {gsm_metrics.get('rlaif_vs_sft_win_rate', 0):.2%}")
        print(f"Verifier-Judge Agreement: {gsm_metrics.get('verifier_judge_agreement', 0):.2%}")
    except Exception as e:
        print("GSM metrics not found.", e)
        
    try:
        svamp_metrics = json.loads(open(results_dir / "transfer_metrics.json").read())
        print("\n--- SVAMP Transfer ---")
        for p in ["sft", "rlvr", "rlaif"]:
            if p in svamp_metrics:
                print(f"{p.upper()}: Acc={svamp_metrics[p]['exact_accuracy']:.2%} Format={svamp_metrics[p]['format_compliance']:.2%} Len={svamp_metrics[p]['mean_length']:.1f}")
        print(f"RLAIF vs SFT Win Rate: {svamp_metrics.get('rlaif_vs_sft_win_rate', 0):.2%}")
        print(f"Verifier-Judge Agreement: {svamp_metrics.get('verifier_judge_agreement', 0):.2%}")
    except Exception as e:
        print("SVAMP metrics not found.", e)
        
    try:
        diag_metrics = json.loads(open(results_dir / "diagnostic_metrics.json").read())
        print("\n--- Diagnostic Set ---")
        print(f"Tie Rate: {diag_metrics.get('tie_rate', 0):.1%}")
        print(f"Wrong Preference Rate: {diag_metrics.get('wrong_preference_rate', 0):.1%}")
        print(f"S_reason Sensitivity: {diag_metrics.get('s_reason_sensitivity', 0):+.1%}")
        print(f"S_outcome Sensitivity: {diag_metrics.get('s_outcome_sensitivity', 0):+.1%}")
    except Exception as e:
        print("Diagnostic metrics not found.", e)

    # 3. Create placeholder for prose
    summary_path = results_dir / "feedback_source_summary.json"
    if not summary_path.exists():
        summary_data = {
            "in_domain_conclusion": "RLVR achieved a higher exact accuracy than RLAIF on the in-domain GSM8K dataset, likely because RLVR was trained directly on the exact answer signal...",
            "transfer_conclusion": "On the SVAMP transfer dataset, both methods generalized differently. RLAIF might show...",
            "diagnostic_conclusion": "The AI judge was easily tricked by persuasive filler and struggled to penalize corrupted reasoning, showing high tie rates when...",
            "final_recommendation": "For verifiably objective tasks like math, RLVR provides a cleaner and more robust training signal, but..."
        }
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
            
    print(f"\nProse summary JSON available at: {summary_path}")
    print("Please edit this file to complete Experiment E.")

if __name__ == "__main__":
    main()
