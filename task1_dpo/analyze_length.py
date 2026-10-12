from __future__ import annotations

import argparse
from common.data import load_yaml, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--eval", action="store_true", help="Also evaluate the length-balanced model after training.")
    ap.add_argument("--skip-train", action="store_true", help="Skip training and only run evaluation.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    if not args.skip_train:
        from task1_dpo.train import run_training
        
        print("Training length-balanced DPO condition...")
        run_training(
            config_path=args.config,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=cfg["length_output"]
        )
        print("Length-balanced training complete.")

    if args.eval or args.skip_train:
        import subprocess
        import sys
        cmd = [
            sys.executable, "-m", "task1_dpo.evaluate",
            "--config", args.config,
            "--adapter", cfg["length_output"],
            "--name", "length_balanced",
            "--eval-dataset", "length_stratified"
        ]
        print(f"Running evaluation: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print("\nTo evaluate on the length-stratified evaluation set, run:")
        print(f"python -m task1_dpo.evaluate --config {args.config} --adapter {cfg['length_output']} --name length_balanced --eval-dataset length_stratified")


if __name__ == "__main__":
    main()
