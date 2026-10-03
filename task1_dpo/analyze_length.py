from __future__ import annotations

import argparse
from common.data import load_yaml, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    from task1_dpo.train import run_training
    
    print("Training length-balanced DPO condition...")
    run_training(
        config_path=args.config,
        run_name="length_balanced",
        dataset_path=cfg["paths"]["dpo_length_train"],
        output_path=cfg["length_output"]
    )
    print("Length-balanced training complete. Make sure to run evaluation on the length-stratified hold-out set.")


if __name__ == "__main__":
    main()
