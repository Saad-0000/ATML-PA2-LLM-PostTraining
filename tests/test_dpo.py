import torch
import unittest
from task1_dpo.dpo import dpo_loss
from common.data import encode_prompt_response, pad_batch
from transformers import AutoTokenizer

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

    def test_causal_token_shifting_pad_batch(self):
        # We can test padding logic directly on synthetic tokens
        # Assuming we have a mock tokenizer
        pass
        

if __name__ == "__main__":
    unittest.main()
