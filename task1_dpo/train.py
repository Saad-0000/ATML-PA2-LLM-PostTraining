from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                c_enc = encode_prompt_response(tokenizer, prompt, yc, max_length)
                r_enc = encode_prompt_response(tokenizer, prompt, yr, max_length)
                chosen.append(c_enc)
                rejected.append(r_enc)
            except ValueError as e:
                # Filter this example if it exceeds max_length
                continue
        
        # If the whole batch gets filtered out
        if not chosen:
            return None, None

        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def get_logprobs(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        response_mask = response_mask.cuda()
        
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    loss_mask = response_mask[:, 1:]
    
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    from common.models import reference_mode
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model = bundle["model"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    beta = bundle["beta"]
    
    grad_acc_steps = int(cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    epochs = int(cfg.get("epochs", 1))
    
    model.train()
    
    step = 0
    for epoch in range(epochs):
        for chosen_batch, rejected_batch in loader:
            if chosen_batch is None:
                continue
                
            with reference_mode(model):
                with torch.no_grad():
                    ref_chosen_logp = get_logprobs(model, chosen_batch)
                    ref_rejected_logp = get_logprobs(model, rejected_batch)
            
            policy_chosen_logp = get_logprobs(model, chosen_batch)
            policy_rejected_logp = get_logprobs(model, rejected_batch)
            
            loss, metrics = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta
            )
            
            loss = loss / grad_acc_steps
            loss.backward()
            
            step += 1
            if step % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch+1} Step {step//grad_acc_steps}, Loss: {loss.item() * grad_acc_steps:.4f}, Acc: {metrics['preference_accuracy']:.4f}")
    
    model.save_pretrained(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
