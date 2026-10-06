from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_k8_cache(path: str) -> dict[str, list[dict]]:
    """Load cached completions grouped by prompt source_index.

    Each prompt in the supplied cache has exactly 8 completions with generation_index 0..7.
    """
    rows = read_jsonl(path)
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    bad = {pid: len(g) for pid, g in by_prompt.items() if len(g) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")

    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return dict(by_prompt)


# ---------------------------------------------------------------------------
# Equal-generation-budget regrouping
# ---------------------------------------------------------------------------

def regroup_equal_generation_budget(
    by_prompt: dict[str, list[dict]], k: int
) -> list[dict[str, Any]]:
    """Partition the 8 completions per prompt into groups of size K.

    Under this equal-budget protocol:
    - K=8: 1 group of 8 per prompt (total generations = 8 * N)
    - K=4: 2 groups of 4 per prompt (completions 0..3 and 4..7; total generations = 8 * N)
    - K=2: 4 groups of 2 per prompt (completions 0..1, 2..3, 4..5, 6..7; total generations = 8 * N)

    This holds the total generation budget strictly constant across K=2, 4, 8.
    """
    groups: list[dict[str, Any]] = []
    group_counter = 0

    for pid, completions in by_prompt.items():
        n_comps = len(completions)
        # Split into consecutive chunks of size k
        for start_idx in range(0, n_comps - (n_comps % k), k):
            subgroup = completions[start_idx : start_idx + k]
            groups.append({
                "group_id": group_counter,
                "prompt_id": str(subgroup[0].get("prompt_id", pid)),
                "source_index": pid,
                "generation_indices": [item.get("generation_index") for item in subgroup],
                "completions": subgroup,
                "rewards": [float(item["reward"]) for item in subgroup],
            })
            group_counter += 1

    return groups


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_group_metrics(groups: list[dict[str, Any]], eps: float = 1e-6) -> dict[str, Any]:
    """Compute the three required quantities for a set of K-sized groups:

    1. informative_group_rate: fraction of groups with reward std > 1e-4
    2. mean_within_group_reward_std: average sample reward std within each group
    3. variance_of_group_relative_signal: variance of relative advantages across all completions
    """
    stds = []
    advantages_all = []
    uninformative_count = 0

    for g in groups:
        r = np.array(g["rewards"], dtype=float)
        std_val = float(np.std(r, ddof=0))
        stds.append(std_val)
        if std_val < 1e-4:
            uninformative_count += 1

        denom = max(std_val, eps)
        adv = (r - float(np.mean(r))) / denom
        advantages_all.extend(adv.tolist())

    n_groups = len(groups)
    inf_rate = float(1.0 - (uninformative_count / max(n_groups, 1)))
    uninf_frac = float(uninformative_count / max(n_groups, 1))
    mean_std = float(np.mean(stds)) if stds else 0.0
    var_signal = float(np.var(advantages_all, ddof=0)) if advantages_all else 0.0

    return {
        "n_groups": n_groups,
        "total_generations": sum(len(g["rewards"]) for g in groups),
        "informative_group_rate": inf_rate,
        "uninformative_group_fraction": uninf_frac,
        "mean_within_group_reward_std": mean_std,
        "variance_of_group_relative_signal": var_signal,
    }


# ---------------------------------------------------------------------------
# Difficulty binning
# ---------------------------------------------------------------------------

def partition_by_difficulty(by_prompt: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, list[dict]], float]:
    """Partition prompts into 'hard' and 'easy' using median prompt mean reward across all 8 cached completions."""
    prompt_means = {
        pid: float(np.mean([item["reward"] for item in comps]))
        for pid, comps in by_prompt.items()
    }
    median_thresh = float(np.median(list(prompt_means.values())))

    hard_prompts = {pid: comps for pid, comps in by_prompt.items() if prompt_means[pid] < median_thresh}
    easy_prompts = {pid: comps for pid, comps in by_prompt.items() if prompt_means[pid] >= median_thresh}

    return hard_prompts, easy_prompts, median_thresh


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_group_size_analysis(config_path: str = "configs/grpo.yaml"):
    cfg = load_yaml(config_path)
    cache_path = cfg["group_cache"]
    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]

    res_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    res_dir.mkdir(parents=True, exist_ok=True)
    out_file = res_dir / "group_size.json"
    qual_file = res_dir / "qualitative_examples.jsonl"

    print(f"Loading K=8 rollout cache from: {cache_path}")
    by_prompt = load_k8_cache(cache_path)
    total_cached_generations = sum(len(v) for v in by_prompt.values())
    print(f"Loaded {len(by_prompt)} prompts, {total_cached_generations} total completions.")

    hard_prompts, easy_prompts, median_threshold = partition_by_difficulty(by_prompt)
    print(f"Difficulty partition (median reward threshold = {median_threshold:.4f}):")
    print(f"  Hard prompts (mean < {median_threshold:.4f}): {len(hard_prompts)}")
    print(f"  Easy prompts (mean >= {median_threshold:.4f}): {len(easy_prompts)}")

    binning_rule = (
        f"Median split on prompt mean reward across all 8 cached completions. "
        f"Prompts with mean reward < {median_threshold:.4f} are binned as 'hard', "
        f"and prompts with mean reward >= {median_threshold:.4f} are binned as 'easy'."
    )

    k_results = {}

    for k in group_sizes:
        groups_all = regroup_equal_generation_budget(by_prompt, k)
        stats_all  = compute_group_metrics(groups_all)

        groups_hard = regroup_equal_generation_budget(hard_prompts, k)
        stats_hard  = compute_group_metrics(groups_hard)

        groups_easy = regroup_equal_generation_budget(easy_prompts, k)
        stats_easy  = compute_group_metrics(groups_easy)

        k_results[f"K={k}"] = {
            "K": k,
            "total_generations_budget": stats_all["total_generations"],
            "overall": stats_all,
            "difficulty_bins": {
                "hard": stats_hard,
                "easy": stats_easy,
            },
        }

        print(
            f"\n[K={k}] Total Budget: {stats_all['total_generations']} completions in {stats_all['n_groups']} groups\n"
            f"  Overall:  informative_rate={stats_all['informative_group_rate']:.4f} | "
            f"within_std={stats_all['mean_within_group_reward_std']:.4f} | "
            f"var_signal={stats_all['variance_of_group_relative_signal']:.4f}\n"
            f"  Hard:     informative_rate={stats_hard['informative_group_rate']:.4f} | "
            f"within_std={stats_hard['mean_within_group_reward_std']:.4f} | "
            f"var_signal={stats_hard['variance_of_group_relative_signal']:.4f}\n"
            f"  Easy:     informative_rate={stats_easy['informative_group_rate']:.4f} | "
            f"within_std={stats_easy['mean_within_group_reward_std']:.4f} | "
            f"var_signal={stats_easy['variance_of_group_relative_signal']:.4f}"
        )

    # Save qualitative examples: one highly informative group, one uninformative group
    groups_k4 = regroup_equal_generation_budget(by_prompt, 4)
    sorted_by_std = sorted(groups_k4, key=lambda g: np.std(g["rewards"], ddof=0))
    uninf_ex = sorted_by_std[0]
    inf_ex = sorted_by_std[-1]

    qual_records = [
        {
            "experimental_condition": "group_size_uninformative_k4",
            "prompt_id": uninf_ex["prompt_id"],
            "prompt_text": f"source_index_{uninf_ex['source_index']}",
            "group_id": uninf_ex["group_id"],
            "K": 4,
            "reward": float(np.mean(uninf_ex["rewards"])),
            "rewards": uninf_ex["rewards"],
            "reward_std": float(np.std(uninf_ex["rewards"], ddof=0)),
            "completion": uninf_ex["completions"][0]["completion"][:300],
            "completion_length": int(uninf_ex["completions"][0]["completion_tokens"]),
            "relevant_statistic": f"group_std={float(np.std(uninf_ex['rewards'], ddof=0)):.6f} (uninformative)",
        },
        {
            "experimental_condition": "group_size_informative_k4",
            "prompt_id": inf_ex["prompt_id"],
            "prompt_text": f"source_index_{inf_ex['source_index']}",
            "group_id": inf_ex["group_id"],
            "K": 4,
            "reward": float(np.mean(inf_ex["rewards"])),
            "rewards": inf_ex["rewards"],
            "reward_std": float(np.std(inf_ex["rewards"], ddof=0)),
            "completion": inf_ex["completions"][0]["completion"][:300],
            "completion_length": int(inf_ex["completions"][0]["completion_tokens"]),
            "relevant_statistic": f"group_std={float(np.std(inf_ex['rewards'], ddof=0)):.4f} (informative)",
        },
    ]

    for rec in qual_records:
        append_jsonl(qual_file, rec)

    # Write output JSON
    output_data = {
        "cache_used": cache_path,
        "total_prompts": len(by_prompt),
        "total_generation_budget": total_cached_generations,
        "difficulty_binning_rule": binning_rule,
        "median_reward_threshold": median_threshold,
        "group_sizes_studied": group_sizes,
        "metric_definitions": {
            "informative_group_rate": "Fraction of groups with within-group reward sample std > 1e-4",
            "mean_within_group_reward_std": "Mean across groups of within-group reward standard deviation ddof=0",
            "variance_of_group_relative_signal": "Sample variance ddof=0 of normalized advantages A_i across all completions",
        },
        "results": k_results,
    }

    save_json(out_file, output_data)
    print(f"\nGroup-size study complete. Results saved to: {out_file}")
    print(f"Qualitative examples appended to: {qual_file}")
    return output_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_group_size_analysis(args.config)


if __name__ == "__main__":
    main()
