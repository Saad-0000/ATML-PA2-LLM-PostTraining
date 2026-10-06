from __future__ import annotations

import argparse
from task3_grpo.normalization import run_normalization_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_normalization_comparison(args.config)


if __name__ == "__main__":
    main()
