from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    betas = cfg["betas"]
    short_ablation_examples = cfg["short_ablation_examples"]
    
    from task1_dpo.train import run_training
    
    for beta in betas:
        run_name = f"beta_{beta}"
        output_path = f"outputs/task1_dpo/{run_name}"
        print(f"Running ablation for beta={beta}, max_examples={short_ablation_examples}")
        run_training(
            config_path=args.config,
            run_name=run_name,
            output_path=output_path,
            beta=beta,
            max_examples=short_ablation_examples
        )


if __name__ == "__main__":
    main()
