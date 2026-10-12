# Task 1 (DPO) Completion Report

## 1. Files Modified or Added
* **Modified:** `task1_dpo/train.py`
* **Modified:** `task1_dpo/evaluate.py`
* **Modified:** `task1_dpo/dpo.py`
* **Modified:** `task1_dpo/analyze_length.py`
* **Modified:** `task1_dpo/qualitative_analysis.py`
* **Modified:** `common/metrics.py`
* **Modified:** `README.md`
* **Added:** `tests/test_dpo.py`

## 2. Bugs Found and Corrections Made
* **Bug 1 (evaluate.py - Word-Limit Evaluation):** 
  * `evaluate_word_limits` failed to extract prompts because it looked for `row["prompt"]` instead of handling the `messages` list structure used in `data/word_limit_prompts.jsonl`.
  * The regex was hardcoded to `in (\d+) words or less`, which matched 0 out of the 10 real prompts (e.g., "at most 40 words", "no more than 25 words", "under 35 words").
  * **Correction:** Replaced with `common.data.prompt_messages` and unified with `common.metrics.parse_word_limit` (with expanded regex patterns). Explicitly tracks skipped prompts and reasons, logs compliance rate in `summary.json`, and records per-example scores.
* **Bug 2 (evaluate.py - Hardcoded Evaluation Dataset):**
  * `evaluate.py` always loaded `dpo_standard_eval.jsonl` even when evaluating the `length_balanced` model, preventing evaluation on `dpo_length_stratified_eval.jsonl`.
  * **Correction:** Added `--eval-dataset` flag (`standard`, `length_stratified`, or custom path) and records dataset metadata in `summary.json`. Also added stratified metrics breakdown by `length_stratum` (`preferred_longer`, `length_matched`, `rejected_longer`).
* **Bug 3 (dpo.py & evaluate.py - Diagnostic Ambiguity):**
  * Preference accuracy was calculated as $(m_\theta > 0)$, while only `policy_margin_mean` was reported in diagnostics, causing confusion between the policy-only margin $\log \pi_\theta(y_w) - \log \pi_\theta(y_l)$ and reference-adjusted DPO margin $m_\theta$.
  * **Correction:** Explicitly separated and reported `policy_margin`, `reference_margin`, `reference_adjusted_margin` ($m_\theta$), `preference_accuracy` ($m_\theta > 0$), and `policy_preference_accuracy` ($\text{policy\_margin} > 0$).
* **Bug 4 (train.py):** Gradient accumulation discarded remaining gradients at the end of the epoch if the batch count was not perfectly divisible by `grad_acc_steps`.
  * **Correction:** Added an `is_last_batch` condition to ensure `optimizer.step()` and `optimizer.zero_grad()` are called on the final partial accumulation window.
* **Bug 5 (train.py):** Lack of `optimizer.zero_grad()` before the training loop starts.
  * **Correction:** Added `optimizer.zero_grad()` before the epoch loop.

## 3. Assignment Requirements Addressed
* **Optimization Loop:** Correct gradient accumulation and optimizer-step counting (Phase 3).
* **Evaluation Pipeline:** Computes held-out DPO loss, preference accuracy ($m_\theta > 0$), policy-only accuracy, margins, response lengths, KL divergence, word-limit compliance, and length-strata breakdowns (Phase 4).
* **Selectable Evaluation Dataset:** Supports `--eval-dataset standard` and `--eval-dataset length_stratified`.
* **Qualitative Analysis:** Identifies standard vs length-balanced differences, word-limit compliance differences, and high-reward/constraint-failure scenarios.
* **Reproducibility:** Dumps an extensive `manifest.json` on training completion and `summary.json` on evaluation completion.

## 4. Tests Run and Status
* `tests/test_dpo.py`:
  * `test_dpo_loss_numerical` (Pass)
  * `test_dpo_loss_gradients` (Pass)
  * `test_dpo_margins_and_preference_correctness` (Pass)
  * `test_word_limit_extraction_all_prompts` (Pass - 10/10 prompts parsed from `word_limit_prompts.jsonl`)
  * `test_word_limit_phrasings_variety` (Pass)
  * `test_messages_prompt_structure_handling` (Pass)
  * `test_word_count_and_compliance` (Pass)
  * `test_eval_dataset_selection` (Pass - validated stratum distribution: exactly 82 per stratum)
  * **Overall:** 8/8 tests passed in 1.195s.

## 5. Full Training Experiments Executed
* **NO**. The full training experiments (standard, length-balanced, and beta ablations) were **not executed** because this current environment lacks the requisite GPU hardware (CUDA unavailable) to reasonably train a 1.5B model in realistic time.

## 6. Paths to Generated Result Artifacts
(These will be generated once you execute the scripts on your Colab GPU instance):
* `outputs/task1_dpo/standard/`
* `outputs/task1_dpo/beta_0.03/`
* `outputs/task1_dpo/beta_0.10/`
* `outputs/task1_dpo/beta_0.30/`
* `outputs/task1_dpo/length_balanced/`
* `results/task1_dpo/standard/`
* `results/task1_dpo/length_balanced/`
* `results/task1_dpo/standard_length_stratified/`
* `results/task1_dpo/qualitative_examples.json`

## 7. Remaining Limitations or Unverified Behavior
* Because GPU memory limitations couldn't be tested here, depending on Colab's VRAM you may need to reduce `batch_size` from 2 to 1 and double `grad_accum_steps` if OOM errors occur during DPO training.

