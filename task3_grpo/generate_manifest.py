"""Generate results/task3_grpo/experiment_manifest.json.

Run AFTER all experiments complete. Reads the saved result files and
populates the manifest with actual values.
"""
from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import load_yaml, repo_path
from common.logging_utils import save_json


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=str(repo_path(".")), stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def get_gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda or "unknown",
            }
    except Exception:
        pass
    return {"device": "cpu", "gpu_name": "none", "cuda_version": "none"}


def get_package_versions() -> dict:
    pkgs = {}
    for pkg in ["torch", "transformers", "peft", "numpy"]:
        try:
            import importlib.metadata
            pkgs[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkgs[pkg] = "unknown"
    return pkgs


def load_result(path: str) -> dict | None:
    p = repo_path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main():
    cfg = load_yaml("configs/grpo.yaml")

    gpu = get_gpu_info()
    pkg = get_package_versions()
    commit = get_git_commit()

    training_summary = load_result("results/task3_grpo/training_summary.json")
    evaluation       = load_result("results/task3_grpo/evaluation.json")
    group_size       = load_result("results/task3_grpo/group_size.json")
    normalization    = load_result("results/task3_grpo/normalization.json")

    manifest = {
        "task":                 "task3_grpo",
        "git_commit":           commit,
        "timestamp":            datetime.datetime.utcnow().isoformat() + "Z",
        "python_version":       platform.python_version(),
        "device":               gpu["device"],
        "gpu_name":             gpu["gpu_name"],
        "cuda_version":         gpu["cuda_version"],
        "packages":             pkg,
        "seed":                 int(cfg["seed"]),
        "model":                cfg["base_model"],
        "midpoint_checkpoint":  cfg["paths"]["grpo_midpoint_policy"],
        "config_file":          "configs/grpo.yaml",
        "config": {
            "updates":                   int(cfg["updates"]),
            "fork_updates":              int(cfg.get("fork_updates", 8)),
            "K":                         int(cfg["num_generations"]),
            "group_sizes":               list(cfg["group_sizes"]),
            "max_prompt_length":         int(cfg["max_prompt_length"]),
            "max_completion_length":     int(cfg["max_completion_length"]),
            "mask_truncated_completions": bool(cfg["mask_truncated_completions"]),
            "learning_rate":             float(cfg["learning_rate"]),
            "clip_epsilon":              float(cfg["clip_epsilon"]),
            "kl_beta":                   float(cfg["kl_beta"]),
            "max_grad_norm":             float(cfg["max_grad_norm"]),
            "policy_epochs":             int(cfg.get("policy_epochs", 1)),
            "seed":                      int(cfg["seed"]),
            "generation": cfg.get("generation", {}),
        },
        "experiments": {
            "standard_grpo": {
                "command":      "python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard",
                "loss_type":    "grpo",
                "updates":      int(cfg["updates"]),
                "K":            int(cfg["num_generations"]),
                "result_file":  "results/task3_grpo/training.jsonl",
                "summary_file": "results/task3_grpo/training_summary.json",
                "output_dir":   str(cfg["output"]),
                "summary":      training_summary.get("config") if training_summary else None,
                "wall_time_sec": training_summary.get("wall_clock_time_sec") if training_summary else None,
                "peak_vram_gb":  training_summary.get("peak_vram_gb") if training_summary else None,
            },
            "heldout_evaluation": {
                "command":     "python -m task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard",
                "adapter":     str(cfg["output"]),
                "result_file": "results/task3_grpo/evaluation.json",
                "summary":     {k: v for k, v in (evaluation or {}).items() if k != "qualitative_examples"},
            },
            "group_size": {
                "command":     "python -m task3_grpo.analyze_group_size --config configs/grpo.yaml",
                "cache":       cfg["group_cache"],
                "group_sizes": list(cfg["group_sizes"]),
                "result_file": "results/task3_grpo/group_size.json",
                "notes":       "Equal-budget: first K completions per prompt from K=8 cache.",
            },
            "normalization": {
                "command":     "python -m task3_grpo.compare_normalization --config configs/grpo.yaml",
                "fork_updates": int(cfg.get("fork_updates", 8)),
                "loss_types":  ["grpo", "dr_grpo"],
                "result_file": "results/task3_grpo/normalization.json",
                "notes":       "Same midpoint, same prompts, same seed, same K.",
            },
        },
        "commands": [
            "python task3_grpo/smoke_test.py",
            "python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard",
            "python -m task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard",
            "python -m task3_grpo.analyze_group_size --config configs/grpo.yaml",
            "python -m task3_grpo.compare_normalization --config configs/grpo.yaml",
            "python task3_grpo/generate_manifest.py",
        ],
        "result_files": {
            "standard_grpo":  "results/task3_grpo/training.jsonl",
            "training_summary": "results/task3_grpo/training_summary.json",
            "evaluation":     "results/task3_grpo/evaluation.json",
            "group_size":     "results/task3_grpo/group_size.json",
            "normalization":  "results/task3_grpo/normalization.json",
            "qualitative":    "results/task3_grpo/normalization.json (per_response_records)",
            "manifest":       "results/task3_grpo/experiment_manifest.json",
        },
        "output_directories": [
            "outputs/task3_grpo/standard",
        ],
        "notes": [
            "max_completion_length reduced from 512 to 256 for T4 GPU efficiency.",
            "All scientific comparisons (K, updates, epsilon, beta) are unchanged.",
            "Group-size study uses supplied cache; no regeneration.",
            "Normalization forks both start from grpo_midpoint_policy checkpoint.",
        ],
    }

    out_path = repo_path("results/task3_grpo/experiment_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, manifest)
    print(f"Manifest written to {out_path}")


if __name__ == "__main__":
    main()
