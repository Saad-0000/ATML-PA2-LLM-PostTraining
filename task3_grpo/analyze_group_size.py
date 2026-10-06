from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_k8_cache(path: str) -> dict[str, list[dict]]:
    """Load cache and group by source_index. Requires >= 8 rows per prompt."""
    rows = read_jsonl(path)
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    bad = {pid: len(g) for pid, g in by_prompt.items() if len(g) < 8}
    if bad:
        raise ValueError(
            f"Expected at least K=8 cached completions per prompt; short groups: {bad}"
        )
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return dict(by_prompt)


# ---------------------------------------------------------------------------
# Regrouping
# ---------------------------------------------------------------------------

def regroup_equal_generation_budget(
    by_prompt: dict[str, list[dict]], k: int
) -> list[list[dict]]:
    """Return K-sized groups while keeping total cached completions fixed.

    Equal-budget rule: we take exactly the first K completions per prompt.
    This ensures every prompt contributes the same number of samples regardless
    of K, so comparisons across K are not confounded by total generation count.

    For K<8 we discard the later generations per prompt; the total number of
    prompts examined stays fixed so the overall generation budget (per prompt)
    is equal across all K values.
    """
    groups: list[list[dict]] = []
    for pid, completions in by_prompt.items():
        # Take the first K in generation order (already sorted)
        groups.append(completions[:k])
    return groups


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------

def compute_group_stats(groups: list[list[dict]], k: int) -> dict[str, Any]:
    """Compute informativeness and signal statistics for a set of K-sized groups."""
    group_stds: list[float] = []
    group_signal_vars: list[float] = []
    is_uninformative: list[bool] = []

    for g in groups:
        rewards = np.array([r["reward"] for r in g], dtype=float)
        std_r = float(np.std(rewards, ddof=0))
        group_stds.append(std_r)
        # Relative advantage variance: std of (r - mu)/max(sigma, eps)
        eps = 1e-6
        advantages = (rewards - rewards.mean()) / max(std_r, eps)
        group_signal_vars.append(float(np.var(advantages, ddof=0)))
        is_uninformative.append(std_r < 1e-4)

    n = len(groups)
    return {
        "K": k,
        "n_groups": n,
        "informative_group_rate": float(1.0 - sum(is_uninformative) / max(n, 1)),
        "uninformative_group_fraction": float(sum(is_uninformative) / max(n, 1)),
        "mean_group_reward_std": float(np.mean(group_stds)) if group_stds else 0.0,
        "std_group_reward_std":  float(np.std(group_stds)) if group_stds else 0.0,
        "mean_signal_variance":  float(np.mean(group_signal_vars)) if group_signal_vars else 0.0,
    }


# ---------------------------------------------------------------------------
# Difficulty binning
# ---------------------------------------------------------------------------

def bin_by_difficulty(
    by_prompt: dict[str, list[dict]], n_bins: int = 2
) -> dict[str, list[list[dict]]]:
    """Bin prompts by mean reward (easy vs hard)."""
    prompt_mean_rewards = {
        pid: float(np.mean([r["reward"] for r in comps]))
        for pid, comps in by_prompt.items()
    }
    sorted_pids = sorted(prompt_mean_rewards, key=lambda p: prompt_mean_rewards[p])
    bin_size = max(1, len(sorted_pids) // n_bins)
    bins: dict[str, list[list[dict]]] = {}
    for b in range(n_bins):
        start = b * bin_size
        end   = start + bin_size if b < n_bins - 1 else len(sorted_pids)
        label = "hard" if b == 0 else "easy"  # low reward = hard
        pids_in_bin = sorted_pids[start:end]
        bins[label] = [by_prompt[p] for p in pids_in_bin]
    return bins


# ---------------------------------------------------------------------------
# Qualitative examples
# ---------------------------------------------------------------------------

def find_qualitative_examples(
    by_prompt: dict[str, list[dict]],
) -> dict[str, Any]:
    """Find one informative and one uninformative group for the report."""
    groups_all = list(by_prompt.values())
    best_informative = max(
        groups_all,
        key=lambda g: np.std([r["reward"] for r in g], ddof=0),
    )
    worst_uninformative = min(
        groups_all,
        key=lambda g: np.std([r["reward"] for r in g], ddof=0),
    )

    def fmt(g):
        rewards = [r["reward"] for r in g]
        return {
            "source_index": g[0]["source_index"],
            "rewards": rewards,
            "reward_std": float(np.std(rewards, ddof=0)),
            "completions": [r.get("response", r.get("completion", ""))[:200] for r in g],
        }

    return {
        "informative_group_example": fmt(best_informative),
        "uninformative_group_example": fmt(worst_uninformative),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    cache_path  = cfg["group_cache"]
    group_sizes = list(cfg["group_sizes"])         # [2, 4, 8]
    res_dir     = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    res_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cache from {cache_path}...")
    by_prompt = load_k8_cache(cache_path)
    print(f"Loaded {len(by_prompt)} prompts. Group sizes to analyze: {group_sizes}")

    # ---- per-K statistics ------------------------------------------------
    results_by_k: dict[str, Any] = {}
    for k in group_sizes:
        groups = regroup_equal_generation_budget(by_prompt, k)
        stats  = compute_group_stats(groups, k)
        results_by_k[f"K={k}"] = stats
        print(
            f"K={k}: informative_rate={stats['informative_group_rate']:.3f} | "
            f"mean_reward_std={stats['mean_group_reward_std']:.4f} | "
            f"mean_signal_var={stats['mean_signal_variance']:.4f} | "
            f"uninform_frac={stats['uninformative_group_fraction']:.3f}"
        )

    # ---- difficulty-binned analysis (for each K) -------------------------
    bins_easy_hard = bin_by_difficulty(by_prompt, n_bins=2)
    difficulty_results: dict[str, Any] = {}
    for k in group_sizes:
        difficulty_results[f"K={k}"] = {}
        for bin_label, bin_groups in bins_easy_hard.items():
            # Regroup: take first K from each prompt's completions in this bin
            truncated_groups = [g[:k] for g in bin_groups]
            stats = compute_group_stats(truncated_groups, k)
            difficulty_results[f"K={k}"][bin_label] = stats
            print(
                f"  K={k} [{bin_label}]: "
                f"informative_rate={stats['informative_group_rate']:.3f} | "
                f"reward_std={stats['mean_group_reward_std']:.4f}"
            )

    # ---- qualitative examples --------------------------------------------
    qual = find_qualitative_examples(by_prompt)
    print("\nInformative group example:")
    print(f"  source={qual['informative_group_example']['source_index']} | "
          f"reward_std={qual['informative_group_example']['reward_std']:.4f} | "
          f"rewards={qual['informative_group_example']['rewards']}")
    print("Uninformative group example:")
    print(f"  source={qual['uninformative_group_example']['source_index']} | "
          f"reward_std={qual['uninformative_group_example']['reward_std']:.6f} | "
          f"rewards={qual['uninformative_group_example']['rewards']}")

    # ---- save results ----------------------------------------------------
    output = {
        "group_size_stats":       results_by_k,
        "difficulty_binned_stats": difficulty_results,
        "qualitative_examples":   qual,
        "notes": [
            "Equal-budget comparison: first K completions per prompt taken from the K=8 cache.",
            "Difficulty bins: lower mean reward = 'hard', higher = 'easy'.",
            "Uninformative group threshold: reward std < 1e-4.",
        ],
    }
    out_path = res_dir / "group_size.json"
    save_json(out_path, output)
    print(f"\nGroup-size analysis saved to {out_path}")


if __name__ == "__main__":
    main()
