import json
from pathlib import Path
import unittest
import torch

from task1_dpo.dpo import dpo_loss
from task1_dpo.evaluate import resolve_eval_dataset
from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.metrics import parse_word_limit, word_count, word_limit_compliance


class TestDPO(unittest.TestCase):
    def test_dpo_loss_numerical(self):
        pi_w = torch.tensor([0.0])
        pi_l = torch.tensor([-1.0])
        ref_w = torch.tensor([0.0])
        ref_l = torch.tensor([0.0])
        beta = 1.0

        # policy_margin = 1.0, ref_margin = 0.0 -> logits = 1.0
        # loss = -logsigmoid(1.0) ~ 0.31326
        loss, metrics = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
        self.assertAlmostEqual(loss.item(), 0.3132616, places=5)
        self.assertEqual(metrics["preference_accuracy"].item(), 1.0)
        self.assertEqual(metrics["policy_preference_accuracy"].item(), 1.0)
        self.assertEqual(metrics["policy_margin_mean"].item(), 1.0)
        self.assertEqual(metrics["reference_margin_mean"].item(), 0.0)
        self.assertEqual(metrics["reference_adjusted_margin_mean"].item(), 1.0)

    def test_dpo_loss_gradients(self):
        pi_w = torch.tensor([0.0], requires_grad=True)
        pi_l = torch.tensor([0.0], requires_grad=True)
        ref_w = torch.tensor([0.0])
        ref_l = torch.tensor([0.0])
        beta = 1.0

        loss, _ = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
        loss.backward()

        # loss = -logsigmoid(beta * (pi_w - pi_l))
        # deriv w.r.t pi_w = -beta * (1 - sigmoid(0)) = -0.5
        self.assertAlmostEqual(pi_w.grad.item(), -0.5)
        self.assertAlmostEqual(pi_l.grad.item(), 0.5)

    def test_dpo_margins_and_preference_correctness(self):
        beta = 0.1

        # Case 1: Both policy and reference agree preferred > rejected, policy margin higher
        pi_w = torch.tensor([2.0])
        pi_l = torch.tensor([0.0])
        ref_w = torch.tensor([1.0])
        ref_l = torch.tensor([0.0])
        loss, metrics = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
        self.assertEqual(metrics["policy_margin_mean"].item(), 2.0)
        self.assertEqual(metrics["reference_margin_mean"].item(), 1.0)
        self.assertEqual(metrics["reference_adjusted_margin_mean"].item(), 1.0)
        self.assertEqual(metrics["preference_accuracy"].item(), 1.0)
        self.assertEqual(metrics["policy_preference_accuracy"].item(), 1.0)

        # Case 2: Policy prefers chosen (margin > 0), but reference had an even stronger preference
        # So reference-adjusted margin m_theta < 0
        pi_w = torch.tensor([1.0])
        pi_l = torch.tensor([0.0])
        ref_w = torch.tensor([3.0])
        ref_l = torch.tensor([0.0])
        loss, metrics = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
        self.assertEqual(metrics["policy_margin_mean"].item(), 1.0)
        self.assertEqual(metrics["reference_margin_mean"].item(), 3.0)
        self.assertEqual(metrics["reference_adjusted_margin_mean"].item(), -2.0)
        self.assertEqual(metrics["preference_accuracy"].item(), 0.0)  # m_theta <= 0
        self.assertEqual(metrics["policy_preference_accuracy"].item(), 1.0)  # policy_margin > 0

        # Case 3: Policy prefers rejected (margin < 0), but reference preferred rejected even more
        # So policy drifted toward chosen relative to reference (m_theta > 0)
        pi_w = torch.tensor([-1.0])
        pi_l = torch.tensor([0.0])
        ref_w = torch.tensor([-3.0])
        ref_l = torch.tensor([0.0])
        loss, metrics = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)
        self.assertEqual(metrics["policy_margin_mean"].item(), -1.0)
        self.assertEqual(metrics["reference_margin_mean"].item(), -3.0)
        self.assertEqual(metrics["reference_adjusted_margin_mean"].item(), 2.0)
        self.assertEqual(metrics["preference_accuracy"].item(), 1.0)  # m_theta > 0
        self.assertEqual(metrics["policy_preference_accuracy"].item(), 0.0)  # policy_margin <= 0

    def test_word_limit_extraction_all_prompts(self):
        prompts_file = repo_path("data/word_limit_prompts.jsonl")
        self.assertTrue(prompts_file.exists())
        rows = read_jsonl(prompts_file)
        self.assertEqual(len(rows), 10)

        expected = {
            "wl01": 40,
            "wl02": 25,
            "wl03": 30,
            "wl04": 35,
            "wl05": 45,
            "wl06": 30,
            "wl07": 35,
            "wl08": 40,
            "wl09": 30,
            "wl10": 45,
        }

        for row in rows:
            pid = row["prompt_id"]
            msgs = prompt_messages(row)
            user_text = msgs[0]["content"]
            limit = parse_word_limit(user_text)
            self.assertIsNotNone(limit, f"Failed to extract word limit from {pid}: {user_text}")
            self.assertEqual(limit, expected[pid], f"Mismatch for {pid}: got {limit}, expected {expected[pid]}")

    def test_word_limit_phrasings_variety(self):
        test_cases = [
            ("Explain regularization in at most 40 words.", 40),
            ("Define overfitting in no more than 25 words.", 25),
            ("Summarize validation sets in under 35 words.", 35),
            ("Answer in at most 30 words.", 30),
            ("Explain gradient descent in 30 words or less.", 30),
            ("Explain attention in 25 words or fewer.", 25),
            ("Describe PCA with maximum 45 words.", 45),
            ("A maximum of 50 words.", 50),
            ("Do not exceed 30 words in your answer.", 30),
            ("Word limit: 20 words.", 20),
            ("Keep your answer within 15 words.", 15),
            ("This prompt has no length constraint whatsoever.", None),
        ]
        for prompt, expected in test_cases:
            res = parse_word_limit(prompt)
            self.assertEqual(res, expected, f"Failed on phrasing '{prompt}': got {res}, expected {expected}")

    def test_messages_prompt_structure_handling(self):
        # 1. messages field with dicts
        row1 = {"prompt_id": "p1", "messages": [{"role": "user", "content": "Explain X in under 30 words."}]}
        msgs1 = prompt_messages(row1)
        self.assertEqual(len(msgs1), 1)
        self.assertEqual(parse_word_limit(msgs1[0]["content"]), 30)

        # 2. prompt field with dicts
        row2 = {"id": "p2", "prompt": [{"role": "user", "content": "Explain Y in at most 20 words."}]}
        msgs2 = prompt_messages(row2)
        self.assertEqual(parse_word_limit(msgs2[0]["content"]), 20)

        # 3. prompt string field
        row3 = {"id": "p3", "prompt": "Explain Z in no more than 15 words."}
        msgs3 = prompt_messages(row3)
        self.assertEqual(parse_word_limit(msgs3[0]["content"]), 15)

        # 4. question field
        row4 = {"id": "p4", "question": "Explain W in 10 words or less."}
        msgs4 = prompt_messages(row4)
        self.assertEqual(parse_word_limit(msgs4[0]["content"]), 10)

    def test_word_count_and_compliance(self):
        text_5 = "One two three four five."
        text_12 = "One two three four five six seven eight nine ten eleven twelve."

        self.assertEqual(word_count(text_5), 5)
        self.assertEqual(word_count(text_12), 12)

        prompt_10 = "Summarize the concept in under 10 words."
        self.assertEqual(word_limit_compliance(prompt_10, text_5), 1.0)
        self.assertEqual(word_limit_compliance(prompt_10, text_12), 0.0)

    def test_eval_dataset_selection(self):
        cfg = load_yaml("configs/dpo.yaml")

        # standard
        name_std, path_std = resolve_eval_dataset(cfg, "standard")
        self.assertEqual(name_std, "standard")
        self.assertTrue(path_std.exists())
        self.assertTrue(str(path_std).endswith("dpo_standard_eval.jsonl"))

        # length_stratified
        name_len, path_len = resolve_eval_dataset(cfg, "length_stratified")
        self.assertEqual(name_len, "length_stratified")
        self.assertTrue(path_len.exists())
        self.assertTrue(str(path_len).endswith("dpo_length_stratified_eval.jsonl"))

        # length alias
        name_alias, path_alias = resolve_eval_dataset(cfg, "length")
        self.assertEqual(name_alias, "length_stratified")
        self.assertEqual(path_alias, path_len)

        # custom path
        name_cust, path_cust = resolve_eval_dataset(cfg, None, data_path="data/word_limit_prompts.jsonl")
        self.assertEqual(name_cust, "custom")
        self.assertTrue(path_cust.exists())

        # Verify length strata balance in dpo_length_stratified_eval.jsonl
        eval_rows = read_jsonl(path_len)
        from collections import Counter
        strata_counts = Counter(r.get("length_stratum") for r in eval_rows)
        self.assertEqual(strata_counts["preferred_longer"], 82)
        self.assertEqual(strata_counts["length_matched"], 82)
        self.assertEqual(strata_counts["rejected_longer"], 82)


if __name__ == "__main__":
    unittest.main()

