"""Numerical/causal controls; all data here are disposable random test fixtures."""

from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

import torch

from swarm.model import BASE_MODELS, CALIBRATORS, ModelConfig, TinyGPT, mixture_log_probabilities, token_loss
from swarm.training import load_checkpoint, optimizer_for, save_checkpoint, set_seed, update


class SwarmCoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        set_seed(17)
        self.config = ModelConfig(2, 32, 4, 64, vocabulary=64, context=8)

    def test_approved_counts_and_distinct_heads(self):
        expected = {"small": 10101760, "medium": 17440512, "small_2": 20283648,
                    "small_4": 40580608, "small_8": 80654400, "medium_2": 34823552,
                    "medium_4": 69868672, "medium_8": 138976640}
        for name, config in {**BASE_MODELS, **CALIBRATORS}.items():
            with torch.device("meta"):
                model = TinyGPT(config)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected[name])
        self.assertNotEqual(BASE_MODELS["small"].width, BASE_MODELS["medium"].width)

    def test_future_tokens_cannot_change_prefix_predictions(self):
        model = TinyGPT(self.config).eval()
        a = torch.randint(64, (2, 8))
        b = a.clone()
        b[:, 4:] = (b[:, 4:] + 7) % 64
        with torch.no_grad():
            torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4], rtol=0, atol=0)

    def test_accumulation_matches_full_batch_even_with_short_last_microbatch(self):
        first = TinyGPT(self.config)
        second = TinyGPT(self.config)
        second.load_state_dict(first.state_dict())
        batch = torch.randint(64, (5, 9))
        a = update(first, optimizer_for(first), batch, microbatch=5)
        b = update(second, optimizer_for(second), batch, microbatch=2)
        self.assertAlmostEqual(a["loss"], b["loss"], places=5)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)

    def test_resume_reproduces_next_update_and_rejects_wrong_identity(self):
        model = TinyGPT(self.config)
        optimizer = optimizer_for(model)
        update(model, optimizer, torch.randint(64, (4, 9)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, progress={"step": 1}, identity={"data": "fixture"})
            batch = torch.randint(64, (4, 9))
            expected = update(model, optimizer, batch)
            resumed, opt, _, progress = load_checkpoint(path, expected_identity={"data": "fixture"})
            actual_batch = torch.randint(64, (4, 9))
            self.assertTrue(torch.equal(batch, actual_batch))
            actual = update(resumed, opt, actual_batch)
            self.assertEqual(expected, actual)
            self.assertEqual(progress["step"], 1)
            for left, right in zip(model.parameters(), resumed.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(path, expected_identity={"data": "different"})

    def test_probability_mixture_and_sparse_zero_weight(self):
        logits = torch.tensor([[[[2., -1.]]], [[[0., 4.]]]])
        expected = 0.25 * logits[0].softmax(-1) + 0.75 * logits[1].softmax(-1)
        actual = mixture_log_probabilities(logits, torch.tensor([0.25, 0.75])).exp()
        torch.testing.assert_close(actual, expected)
        sparse = mixture_log_probabilities(logits, torch.tensor([1., 0.]))
        torch.testing.assert_close(sparse, logits[0].log_softmax(-1))

    def test_suffix_loss_excludes_prefix_and_finite_training_updates(self):
        logits = torch.zeros(2, 8, 64)
        targets = torch.zeros(2, 8, dtype=torch.long)
        logits[:, :4, 0] = -1000
        total, count = token_loss(logits, targets, score_from=4)
        self.assertEqual(count, 8)
        self.assertAlmostEqual(float(total / count), float(torch.tensor(64.).log()), places=5)
        model = TinyGPT(self.config)
        optimizer = optimizer_for(model, learning_rate=0.005)
        fixture = torch.ones(4, 9, dtype=torch.long)
        losses = [update(model, optimizer, fixture)["loss"] for _ in range(8)]
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
