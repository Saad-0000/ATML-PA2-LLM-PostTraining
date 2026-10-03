from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    kl_values = cfg["kl_values"]
    fork_updates = cfg["fork_updates"]
    
    from task2_ppo.continue_train import run_ppo
    for beta in kl_values:
        run_name = f"kl_{beta}"
        output_path = f"outputs/task2_ppo/{run_name}"
        print(f"Running ablation for kl_beta={beta}, updates={fork_updates}")
        run_ppo(
            config_path=args.config,
            output=output_path,
            updates=fork_updates,
            kl_beta=beta,
            run_name=run_name
        )


if __name__ == "__main__":
    main()
