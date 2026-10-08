from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    rows = read_jsonl(src)
    ids = fixed_audit_ids(rows, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    
    # Extract the actual prompt and response for these IDs so they are readable in the CSV
    audit_data = []
    for r in rows:
        if r["xstest_id"] in ids:
            audit_data.append({
                "xstest_id": r["xstest_id"],
                "prompt": r["prompt"],
                "sft_response": r["response"],
                "manual_label": ""
            })
            
    pd.DataFrame(audit_data).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs with text:", outdir / "manual_audit_ids.csv")


if __name__ == "__main__":
    main()
