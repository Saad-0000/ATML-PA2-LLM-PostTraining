from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, repo_path


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    clip_values = cfg["clip_values"]
    fork_updates = cfg["fork_updates"]

    import json
    from task2_ppo.ppo import ppo_policy_loss
    from task2_ppo.continue_train import run_ppo
    
    print(f"Loaded {len(rows)} cached rollouts for clipping analysis.")
    cached_results = {}
    
    for eps in clip_values:
        total_affected = 0
        total_tokens = 0
        
        for r in rows:
            old_lp = torch.tensor(r["old_logprobs"])
            ref_lp = torch.tensor(r["ref_logprobs"])
            ratio = torch.exp(old_lp - ref_lp)
            affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
            total_affected += affected.sum().item()
            total_tokens += affected.numel()
            
        affected_frac = total_affected / max(1, total_tokens)
        cached_results[f"eps_{eps}"] = {
            "epsilon": eps,
            "affected_token_fraction": affected_frac,
            "total_tokens": total_tokens,
        }
        print(f"Cached Rollouts [epsilon={eps}]: Affected Token Fraction = {affected_frac:.4f}")

    res_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    res_dir.mkdir(parents=True, exist_ok=True)
    cache_json_path = res_dir / "cached_clipping_analysis.json"
    cache_json_path.write_text(json.dumps(cached_results, indent=2))
    print(f"Saved cached rollout clipping stats to {cache_json_path}")
    
    for eps in clip_values:
        run_name = f"clip_{eps}"
        output_path = f"outputs/task2_ppo/{run_name}"
        print(f"\n--- Running short fork for epsilon={eps} ---")
        run_ppo(
            config_path=args.config,
            output=output_path,
            updates=fork_updates,
            clip_epsilon=eps,
            run_name=run_name
        )


if __name__ == "__main__":
    main()
