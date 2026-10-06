"""Audit script for Task 3 GRPO evidence.

Verifies that all required files and metrics exist and conform to assignment specifications:
- Standard GRPO continuation (20 updates, K=4, all required metrics)
- Group-size study (K=2, 4, 8, equal budget, informative rate, reward std, signal variance, difficulty bins)
- Normalization comparison (canonical vs Dr. GRPO, held-out metrics, length-conditioned statistics)
- Qualitative examples (qualitative_examples.jsonl exists and non-empty)
- Reproducibility manifest (experiment_manifest.json exists with actual configuration)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.data import repo_path


def audit_task3_evidence() -> bool:
    print("=" * 65)
    print("TASK 3 GRPO EVIDENCE AUDIT")
    print("=" * 65)

    all_passed = True
    res_dir = repo_path("results/task3_grpo")

    def check(name: str, condition: bool, details: str = ""):
        nonlocal all_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_passed = False
        print(f"[{status}] {name}" + (f" -> {details}" if details else ""))

    # -----------------------------------------------------------------------
    # 1. Standard GRPO Continuation
    # -----------------------------------------------------------------------
    print("\n--- 1. Standard GRPO Continuation ---")
    train_summary_file = res_dir / "training_summary.json"
    train_jsonl_file = res_dir / "training.jsonl"

    if not train_summary_file.exists():
        check("training_summary.json exists", False, f"Missing: {train_summary_file}")
    else:
        check("training_summary.json exists", True)
        try:
            summary = json.loads(train_summary_file.read_text(encoding="utf-8"))
            cfg = summary.get("config", {})
            history = summary.get("history", [])

            check("20 updates recorded", len(history) == 20 or cfg.get("updates") == 20, f"Found {len(history)} updates")
            check("K=4 generations", int(cfg.get("K", 0)) == 4, f"K={cfg.get('K')}")
            check("max_completion_length=512", int(cfg.get("max_completion_length", 0)) == 512, f"Length={cfg.get('max_completion_length')}")

            if history:
                first = history[0]
                req_keys = [
                    "mean_reward", "reward_std", "mean_within_group_reward_std",
                    "uninformative_group_fraction", "policy_loss", "policy_term",
                    "kl", "entropy", "grad_norm", "mean_response_length",
                    "peak_vram_gb", "elapsed_sec", "truncation_count", "truncation_rate"
                ]
                for k in req_keys:
                    check(f"Metric '{k}' present in updates", k in first, f"sample value: {first.get(k)}")
        except Exception as e:
            check("Parse training_summary.json", False, str(e))

    check("training.jsonl exists", train_jsonl_file.exists())

    # -----------------------------------------------------------------------
    # 2. Group-Size Study
    # -----------------------------------------------------------------------
    print("\n--- 2. Group-Size Study ---")
    group_file = res_dir / "group_size.json"
    if not group_file.exists():
        check("group_size.json exists", False, f"Missing: {group_file}")
    else:
        check("group_size.json exists", True)
        try:
            gdata = json.loads(group_file.read_text(encoding="utf-8"))
            check("Cache provenance recorded", bool(gdata.get("cache_used")), gdata.get("cache_used", ""))
            check("Total generation budget recorded", int(gdata.get("total_generation_budget", 0)) == 192, f"budget={gdata.get('total_generation_budget')}")
            check("Difficulty binning rule defined", bool(gdata.get("difficulty_binning_rule")), gdata.get("difficulty_binning_rule", "")[:60] + "...")

            res = gdata.get("results", {})
            check("K=2 studied", "K=2" in res)
            check("K=4 studied", "K=4" in res)
            check("K=8 studied", "K=8" in res)

            for k_str in ["K=2", "K=4", "K=8"]:
                if k_str in res:
                    kr = res[k_str]
                    ov = kr.get("overall", {})
                    diff = kr.get("difficulty_bins", {})
                    check(f"{k_str} equal generation budget", int(kr.get("total_generations_budget", 0)) == 192)
                    check(f"{k_str} informative_group_rate present", "informative_group_rate" in ov)
                    check(f"{k_str} within_group_reward_std present", "mean_within_group_reward_std" in ov)
                    check(f"{k_str} variance_of_group_relative_signal present", "variance_of_group_relative_signal" in ov)
                    check(f"{k_str} hard & easy bins present", "hard" in diff and "easy" in diff)
        except Exception as e:
            check("Parse group_size.json", False, str(e))

    # -----------------------------------------------------------------------
    # 3. Normalization Study
    # -----------------------------------------------------------------------
    print("\n--- 3. Normalization Study ---")
    norm_file = res_dir / "normalization.json"
    if not norm_file.exists():
        check("normalization.json exists", False, f"Missing: {norm_file}")
    else:
        check("normalization.json exists", True)
        try:
            ndata = json.loads(norm_file.read_text(encoding="utf-8"))
            ctrls = ndata.get("experimental_controls", {})
            heldout = ndata.get("heldout_comparison", {})
            lc = ndata.get("length_conditioned_analysis", {})

            check("Fixed seed recorded", "seed" in ctrls, str(ctrls.get("seed")))
            check("Fixed beta and epsilon recorded", "kl_beta" in ctrls and "clip_epsilon" in ctrls)
            check("Canonical GRPO heldout present", "canonical_grpo" in heldout)
            check("Dr.-GRPO heldout present", "dr_grpo" in heldout)
            check("Length-conditioned analysis present", "canonical_grpo" in lc and "dr_grpo" in lc)

            for cond in ["canonical_grpo", "dr_grpo"]:
                if cond in lc:
                    cdata = lc[cond]
                    check(f"{cond} short/long bins defined", "short_completions" in cdata and "long_completions" in cdata)
        except Exception as e:
            check("Parse normalization.json", False, str(e))

    # -----------------------------------------------------------------------
    # 4. Qualitative Examples
    # -----------------------------------------------------------------------
    print("\n--- 4. Qualitative Examples ---")
    qual_file = res_dir / "qualitative_examples.jsonl"
    if not qual_file.exists():
        check("qualitative_examples.jsonl exists", False, f"Missing: {qual_file}")
    else:
        lines = [l for l in qual_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        check("qualitative_examples.jsonl non-empty", len(lines) > 0, f"{len(lines)} examples saved")
        if lines:
            first_q = json.loads(lines[0])
            check("Example has completion, reward, condition", "completion" in first_q and "reward" in first_q and "experimental_condition" in first_q)

    # -----------------------------------------------------------------------
    # 5. Reproducibility Manifest
    # -----------------------------------------------------------------------
    print("\n--- 5. Reproducibility Manifest ---")
    manifest_file = res_dir / "experiment_manifest.json"
    if not manifest_file.exists():
        check("experiment_manifest.json exists", False, f"Missing: {manifest_file}")
    else:
        check("experiment_manifest.json exists", True)
        try:
            mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
            check("Task name is task3_grpo", mdata.get("task") == "task3_grpo")
            check("Commands recorded", bool(mdata.get("commands")))
            check("Result files mapped", bool(mdata.get("result_files")))
        except Exception as e:
            check("Parse experiment_manifest.json", False, str(e))

    print("\n" + "=" * 65)
    if all_passed:
        print("ALL TASK 3 EVIDENCE CHECKS PASSED SUCCESSFULLY!")
    else:
        print("SOME CHECKS FAILED OR RESULT FILES ARE PENDING RUNTIME COMPLETION.")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    success = audit_task3_evidence()
    # Exit with code 0 if all pass, code 1 if missing evidence
    sys.exit(0 if success else 1)
