from __future__ import annotations

import torch
from common.data import load_yaml
from common.models import load_value_model, token_values

def main():
    cfg = load_yaml("configs/ppo.yaml")
    print("Loading value model...")
    vm = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode=cfg.get("value_train_mode", "head_only"))
    
    seq = torch.randint(0, 1000, (1, 16))
    attn = torch.ones((1, 16))
    if torch.cuda.is_available():
        seq = seq.cuda()
        attn = attn.cuda()
        vm = vm.cuda()

    print("\n--- Testing token_values ---")
    vals = token_values(vm, seq, attn).float()
    print("vals shape:", vals.shape, "dtype:", vals.dtype, "sample:", vals[0, :4])
    assert torch.isfinite(vals).all(), "vals contains non-finite values!"
    print("SUCCESS: token_values runs cleanly and returns float32 finite values!")

if __name__ == "__main__":
    main()
