from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.metrics import (
    parse_word_limit,
    sampled_kl,
    word_count,
    word_limit_compliance,
)
from common.models import load_policy, load_reward_model, load_tokenizer
from task1_dpo.dpo import dpo_loss


def log_status(msg: str) -> None:
    """Print timestamped status message and immediately flush stdout for live visibility."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def resolve_eval_dataset(cfg: dict, dataset_name: str | None, data_path: str | None = None) -> tuple[str, Path]:
    if data_path:
        p = repo_path(data_path)
        return "custom", p
    if dataset_name in ("length_stratified", "length", "length_balanced"):
        return "length_stratified", repo_path(cfg["paths"]["dpo_length_eval"])
    return "standard", repo_path(cfg["paths"]["dpo_standard_eval"])


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
    eval_dataset: str | None = None,
    eval_data_path: str | None = None,
):
    log_status(f"[1/5: Model Loading] Reading configuration from {config_path}...")
    cfg = load_yaml(config_path)
    ds_name, ds_path = resolve_eval_dataset(cfg, eval_dataset, eval_data_path)

    log_status(f"[1/5: Model Loading] Reading evaluation dataset '{ds_name}' from {ds_path}...")
    rows = read_jsonl(ds_path)
    word_limit_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    log_status(
        f"[1/5: Model Loading] Loaded {len(rows)} preference pairs and {len(word_limit_rows)} word-limit prompts."
    )

    log_status(f"[1/5: Model Loading] Initializing tokenizer ({cfg['base_model']})...")
    tokenizer = load_tokenizer(cfg["base_model"])

    log_status(f"[1/5: Model Loading] Loading evaluation policy with adapter ({adapter})...")
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)

    log_status(f"[1/5: Model Loading] Loading frozen reference policy ({cfg['base_model']})...")
    ref_policy = load_policy(cfg, adapter_path=None, trainable=False)

    log_status(f"[1/5: Model Loading] Loading reward model ({cfg['reward_model']})...")
    reward = load_reward_model(cfg)

    log_status("[1/5: Model Loading] All models and tokenizers loaded successfully.")
    return {
        "cfg": cfg,
        "eval_dataset_name": ds_name,
        "eval_dataset_path": str(ds_path),
        "rows": rows,
        "word_limit_rows": word_limit_rows,
        "tokenizer": tokenizer,
        "policy": policy,
        "ref_policy": ref_policy,
        "reward": reward,
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


def get_per_token_logprobs(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return per_token_logps


def evaluate_preferences(policy, ref_policy, tokenizer, rows, cfg):
    max_length = int(cfg["max_sequence_length"])
    beta = float(cfg["beta"])

    results = []
    skipped_count = 0

    pbar = tqdm(
        rows,
        desc="[2/5 Preference Eval]",
        total=len(rows),
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for row in pbar:
        prompt = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)

        try:
            c_enc = encode_prompt_response(tokenizer, prompt, yc, max_length)
            r_enc = encode_prompt_response(tokenizer, prompt, yr, max_length)
        except ValueError:
            skipped_count += 1
            pbar.set_postfix({"skipped": skipped_count})
            continue

        c_batch = pad_batch(tokenizer, [c_enc])
        r_batch = pad_batch(tokenizer, [r_enc])

        with torch.no_grad():
            ref_chosen_logp = get_logprobs(ref_policy, c_batch)
            ref_rejected_logp = get_logprobs(ref_policy, r_batch)

            pol_chosen_logp = get_logprobs(policy, c_batch)
            pol_rejected_logp = get_logprobs(policy, r_batch)

            loss, metrics = dpo_loss(
                pol_chosen_logp, pol_rejected_logp,
                ref_chosen_logp, ref_rejected_logp,
                beta
            )

        is_correct = bool(metrics["preference_accuracy"].item() > 0)
        policy_correct = bool(metrics["policy_preference_accuracy"].item() > 0)

        results.append({
            "id": row.get("prompt_id") or row.get("id", ""),
            "category": row.get("length_stratum", row.get("category", "unlabeled")),
            "length_stratum": row.get("length_stratum", ""),
            "loss": float(loss.item()),
            "preference_correct": is_correct,
            "policy_preference_correct": policy_correct,
            "policy_margin": float(metrics["policy_margin_mean"].item()),
            "reference_margin": float(metrics["reference_margin_mean"].item()),
            "reference_adjusted_margin": float(metrics["reference_adjusted_margin_mean"].item()),
            "dpo_margin": float(metrics["dpo_margin_mean"].item()),
            "chosen_length": word_count(yc),
            "rejected_length": word_count(yr),
        })

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "m_theta": f"{metrics['reference_adjusted_margin_mean'].item():.2f}",
            "skipped": skipped_count,
        })

    log_status(
        f"[2/5: Preference Evaluation] Completed: {len(results)} evaluated, "
        f"{skipped_count} skipped (exceeded max_sequence_length={max_length})."
    )
    return results


def evaluate_generation(policy, ref_policy, reward_bundle, tokenizer, rows, cfg):
    reward_model, r_tokenizer = reward_bundle
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen = int(cfg.get("max_generation_tokens", 256))

    results = []
    skipped_count = 0

    pbar = tqdm(
        rows,
        desc="[3/5 Generation Eval]",
        total=len(rows),
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for row in pbar:
        prompt = prompt_messages_from_preference(row)
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) > max_prompt:
            skipped_count += 1
            pbar.set_postfix({"skipped": skipped_count})
            continue

        input_ids = torch.tensor([prompt_ids]).to(policy.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = policy.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        gen_ids = outputs[0][len(prompt_ids):]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        response_len = len(gen_ids)

        # Reward
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rm_inputs = r_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1280).to(reward_model.device)
        with torch.no_grad():
            score = float(reward_model(**rm_inputs).logits[0, 0].item())

        # KL calculation
        seq = outputs
        attn = torch.ones_like(seq)
        resp_mask = torch.zeros_like(seq)[:, 1:]
        resp_mask[0, len(prompt_ids)-1:] = 1

        batch = {"input_ids": seq, "attention_mask": attn}
        with torch.no_grad():
            ref_logp = get_per_token_logprobs(ref_policy, batch)
            pol_logp = get_per_token_logprobs(policy, batch)
            kl = float(sampled_kl(pol_logp, ref_logp, resp_mask).item())

        results.append({
            "id": row.get("prompt_id") or row.get("id", ""),
            "category": row.get("length_stratum", row.get("category", "unlabeled")),
            "length_stratum": row.get("length_stratum", ""),
            "prompt": prompt,
            "response": response_text,
            "reward": score,
            "kl": kl,
            "length": response_len,
            "word_count": word_count(response_text),
        })

        pbar.set_postfix({
            "rew": f"{score:.2f}",
            "kl": f"{kl:.2f}",
            "len": response_len,
            "skipped": skipped_count,
        })

    log_status(
        f"[3/5: Generation Evaluation] Completed: {len(results)} evaluated, "
        f"{skipped_count} skipped (prompt length > {max_prompt})."
    )
    return results


def evaluate_word_limits(policy, reward_bundle, tokenizer, rows, cfg):
    reward_model, r_tokenizer = reward_bundle
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_gen = int(cfg.get("max_generation_tokens", 256))

    results = []
    skipped = []

    pbar = tqdm(
        rows,
        desc="[4/5 Word Limit Eval]",
        total=len(rows),
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for row in pbar:
        row_id = row.get("prompt_id") or row.get("id", "")
        messages = prompt_messages(row)

        prompt_text = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                prompt_text = str(m.get("content", ""))
                break
        if not prompt_text and messages and isinstance(messages[-1], dict):
            prompt_text = str(messages[-1].get("content", ""))
        if not prompt_text and isinstance(row.get("prompt"), str):
            prompt_text = row["prompt"]

        if not prompt_text.strip():
            skipped.append({
                "id": row_id,
                "reason": "empty_prompt",
                "details": "Could not extract user prompt text from row.",
            })
            pbar.set_postfix({"skipped": len(skipped)})
            continue

        limit = parse_word_limit(prompt_text)
        if limit is None:
            skipped.append({
                "id": row_id,
                "reason": "unrecognized_word_limit",
                "prompt": prompt_text,
                "details": "No recognized word limit constraint found in prompt text.",
            })
            pbar.set_postfix({"skipped": len(skipped)})
            continue

        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(prompt_ids) > max_prompt:
            skipped.append({
                "id": row_id,
                "reason": "prompt_too_long",
                "details": f"Prompt length {len(prompt_ids)} exceeds max_prompt {max_prompt}.",
            })
            pbar.set_postfix({"skipped": len(skipped)})
            continue

        input_ids = torch.tensor([prompt_ids]).to(policy.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = policy.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        gen_ids = outputs[0][len(prompt_ids):]
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        words = word_count(response_text)
        is_compliant = bool(words <= limit)

        # Also score with reward model to enable high-reward vs failed-constraint qualitative analysis
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rm_inputs = r_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1280).to(reward_model.device)
        with torch.no_grad():
            score = float(reward_model(**rm_inputs).logits[0, 0].item())

        results.append({
            "id": row_id,
            "prompt": prompt_text,
            "response": response_text,
            "limit": limit,
            "word_count": words,
            "compliant": is_compliant,
            "reward": score,
            "length": len(gen_ids),
        })

        pbar.set_postfix({
            "limit": limit,
            "words": words,
            "ok": is_compliant,
            "skipped": len(skipped),
        })

    log_status(
        f"[4/5: Word Limit Evaluation] Completed: {len(results)} evaluated, "
        f"{len(skipped)} skipped."
    )
    return results, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument(
        "--eval-dataset",
        choices=["standard", "length_stratified", "length"],
        default=None,
        help="Evaluation dataset: 'standard' (dpo_standard_eval) or 'length_stratified' (dpo_length_eval).",
    )
    ap.add_argument("--eval-data-path", default=None, help="Explicit path to evaluation JSONL file.")
    args = ap.parse_args()

    # Determine evaluation dataset explicitly, preserving backward compatibility
    eval_dataset = args.eval_dataset
    if eval_dataset is None:
        if "length" in args.name.lower() or "length" in str(args.adapter).lower():
            eval_dataset = "length_stratified"
            log_status(f"Notice: Selecting --eval-dataset 'length_stratified' based on adapter/name: {args.name}")
        else:
            eval_dataset = "standard"

    log_status(
        f"=== Starting Task 1 Evaluation: run='{args.name}', adapter='{args.adapter}', eval_dataset='{eval_dataset}' ==="
    )

    # Stage 1: Load Models and Datasets
    bundle = load_evaluation_bundle(
        args.config,
        args.adapter,
        eval_dataset=eval_dataset,
        eval_data_path=args.eval_data_path,
    )
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    ref_policy = bundle["ref_policy"]
    tokenizer = bundle["tokenizer"]

    output_dir = repo_path(cfg["results_dir"]) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 2: Preference Accuracy and DPO Loss Evaluation
    log_status(
        f"[2/5: Preference Evaluation] Evaluating {len(bundle['rows'])} pairs on '{bundle['eval_dataset_name']}'..."
    )
    pref_results = evaluate_preferences(policy, ref_policy, tokenizer, bundle["rows"], cfg)
    pref_out = output_dir / "preference_results.jsonl"
    with open(pref_out, "w") as f:
        for r in pref_results:
            f.write(json.dumps(r) + "\n")
    log_status(f"[2/5: Preference Evaluation] Wrote {len(pref_results)} records to {pref_out}")

    # Stage 3: Generation Evaluation
    log_status(
        f"[3/5: Generation Evaluation] Evaluating generation on '{bundle['eval_dataset_name']}' ({len(bundle['rows'])} prompts)..."
    )
    gen_results = evaluate_generation(policy, ref_policy, bundle["reward"], tokenizer, bundle["rows"], cfg)
    gen_out = output_dir / "generation_results.jsonl"
    with open(gen_out, "w") as f:
        for r in gen_results:
            f.write(json.dumps(r) + "\n")
    log_status(f"[3/5: Generation Evaluation] Wrote {len(gen_results)} records to {gen_out}")

    # Stage 4: Word Limits Evaluation
    log_status(
        f"[4/5: Word Limit Evaluation] Evaluating {len(bundle['word_limit_rows'])} word-limit prompts..."
    )
    word_results, word_skipped = evaluate_word_limits(
        policy, bundle["reward"], tokenizer, bundle["word_limit_rows"], cfg
    )
    word_out = output_dir / "word_limit_results.jsonl"
    with open(word_out, "w") as f:
        for r in word_results:
            f.write(json.dumps(r) + "\n")
    log_status(f"[4/5: Word Limit Evaluation] Wrote {len(word_results)} records to {word_out}")

    if word_skipped:
        skip_out = output_dir / "word_limit_skipped.jsonl"
        with open(skip_out, "w") as f:
            for s in word_skipped:
                f.write(json.dumps(s) + "\n")
        log_status(f"[4/5: Word Limit Evaluation] Wrote {len(word_skipped)} skipped records to {skip_out}")

    # Stage 5: Aggregate metrics and save summary
    log_status("[5/5: Saving Results] Aggregating summary metrics...")
    summary = {
        "run_name": args.name,
        "adapter": str(args.adapter),
        "eval_dataset": bundle["eval_dataset_name"],
        "eval_dataset_path": bundle["eval_dataset_path"],
        "n_pref_eval": len(pref_results),
        "n_gen_eval": len(gen_results),
        "n_word_eval": len(word_results),
        "n_word_skipped": len(word_skipped),
    }

    if pref_results:
        summary["dpo_loss"] = float(np.mean([r["loss"] for r in pref_results]))
        summary["preference_accuracy"] = float(np.mean([r["preference_correct"] for r in pref_results]))
        summary["policy_preference_accuracy"] = float(np.mean([r["policy_preference_correct"] for r in pref_results]))
        summary["policy_margin_mean"] = float(np.mean([r["policy_margin"] for r in pref_results]))
        summary["reference_margin_mean"] = float(np.mean([r["reference_margin"] for r in pref_results]))
        summary["reference_adjusted_margin_mean"] = float(np.mean([r["reference_adjusted_margin"] for r in pref_results]))
        summary["dpo_margin_mean"] = summary["reference_adjusted_margin_mean"]

        # Stratified analysis by length_stratum (or category)
        strata = defaultdict(list)
        for r in pref_results:
            key = r.get("length_stratum") or r.get("category") or "unlabeled"
            strata[key].append(r)

        summary["strata_preference_accuracy"] = {
            s: float(np.mean([r["preference_correct"] for r in items]))
            for s, items in strata.items()
        }
        summary["strata_policy_margin"] = {
            s: float(np.mean([r["policy_margin"] for r in items]))
            for s, items in strata.items()
        }
        summary["strata_ref_adjusted_margin"] = {
            s: float(np.mean([r["reference_adjusted_margin"] for r in items]))
            for s, items in strata.items()
        }
        # Backward compatibility alias
        summary["stratified_pref_acc"] = summary["strata_preference_accuracy"]

    if gen_results:
        summary["reward_mean"] = float(np.mean([r["reward"] for r in gen_results]))
        summary["kl_mean"] = float(np.mean([r["kl"] for r in gen_results]))
        summary["length_mean"] = float(np.mean([r["length"] for r in gen_results]))
        summary["length_std"] = float(np.std([r["length"] for r in gen_results]))

        strata_gen = defaultdict(list)
        for r in gen_results:
            key = r.get("length_stratum") or r.get("category") or "unlabeled"
            strata_gen[key].append(r)

        summary["strata_reward"] = {
            s: float(np.mean([r["reward"] for r in items]))
            for s, items in strata_gen.items()
        }
        summary["strata_length"] = {
            s: float(np.mean([r["length"] for r in items]))
            for s, items in strata_gen.items()
        }
        # Backward compatibility aliases
        summary["stratified_reward"] = summary["strata_reward"]
        summary["stratified_length"] = summary["strata_length"]

    if word_results:
        summary["word_limit_compliance"] = float(np.mean([r["compliant"] for r in word_results]))
    if word_skipped:
        summary["word_limit_skipped_reasons"] = [s["reason"] for s in word_skipped]

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    log_status(f"[5/5: Saving Results] Saved summary JSON to {summary_file}")
    log_status(f"=== Completed Task 1 Evaluation for '{args.name}' ===")

    print("\n--- Summary ---", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
