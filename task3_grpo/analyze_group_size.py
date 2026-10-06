from __future__ import annotations

import argparse
from task3_grpo.group_size import run_group_size_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_group_size_analysis(args.config)


if __name__ == "__main__":
    main()
