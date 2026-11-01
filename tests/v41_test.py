"""Focused R-Build-v4.1 acceptance tests (standard-library unittest)."""

import itertools
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from rbuild import (
    RBuildAdaptedModel,
    RBuildBaseTrainer,
    V41Config,
    continued_pretraining_batches,
    supervised_fine_tuning_batches,
    v41_release_gate,
)


class ToyTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(char) % 20) for char in text]


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        logits = self.head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, input_ids[:, -1:]], dim=1)


def tiny_v41(memory=True, critics=True):
    config = V41Config(
        memory_enabled=memory,
        critic_enabled=critics,
        n_critics=2,
        y_critics=1,
        critic_hidden=8,
    )
    config.train.max_steps = 1
    config.train.warmup_steps = 1
    config.train.grad_accum = 1
    config.train.precision = "fp32"
    config.train.lr = 1e-3
    config.train.adamw_lr = 1e-3
    return config


class V41Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.ids = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)

    def test_any_base_exact_passthrough_and_no_surgery(self):
        base = ToyCausalLM().eval()
        before = {name: value.detach().clone() for name, value in base.state_dict().items()}
        expected = base(self.ids, labels=self.ids)
        model = RBuildAdaptedModel(base, tiny_v41()).eval()
        actual = model(self.ids, labels=self.ids)
        self.assertTrue(torch.equal(expected.logits, actual.logits))
        self.assertTrue(torch.equal(expected.loss, actual.loss))
        self.assertTrue(torch.equal(base.generate(self.ids), model.generate(self.ids)))
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        gate = v41_release_gate(expected, actual, 1.0, 1.0)
        self.assertTrue(gate["passed"])

    def test_memory_critic_write_and_checkpoint(self):
        model = RBuildAdaptedModel(ToyCausalLM(), tiny_v41())
        hidden = torch.randn(2, 3, 16)
        written = model.remember_hidden(hidden, require_critic=False)
        self.assertEqual(written, 6)
        recalled = model.recall_hidden(hidden)
        self.assertEqual(recalled.shape, hidden.shape)
        with tempfile.TemporaryDirectory() as path:
            model.save_checkpoint(path)
            restored = RBuildAdaptedModel.load_checkpoint(path, base_model=ToyCausalLM())
            self.assertTrue(torch.equal(model.memory.matrix, restored.memory.matrix))
            for left, right in zip(model.base_model.parameters(), restored.base_model.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_continued_pretraining_batches(self):
        batches = continued_pretraining_batches(
            ToyTokenizer(), ["abcdefghij", "klmnopqrst"], seq_len=8, batch_size=2
        )
        batch = next(batches)
        self.assertEqual(batch["input_ids"].shape, (2, 8))
        self.assertTrue(torch.equal(batch["input_ids"], batch["labels"]))
        self.assertTrue(bool((batch["attention_mask"] == 1).all()))

    def test_supervised_fine_tuning_masks_prompt(self):
        examples = [{"prompt": "ab", "response": "cde"}]
        batch = next(
            supervised_fine_tuning_batches(
                ToyTokenizer(), examples, seq_len=8, batch_size=1, shuffle=False
            )
        )
        labels = batch["labels"][0]
        self.assertTrue(bool((labels[:2] == -100).all()))
        self.assertTrue(bool((labels[2:6] != -100).all()))
        self.assertTrue(bool((labels[6:] == -100).all()))
        self.assertEqual(int(batch["attention_mask"].sum()), 6)

    def test_continued_pretraining_and_fine_tuning_step(self):
        for batches in (
            continued_pretraining_batches(ToyTokenizer(), ["abcdefghij" * 4], seq_len=8),
            supervised_fine_tuning_batches(
                ToyTokenizer(),
                [{"prompt": "ab", "response": "cdefgh"}],
                seq_len=8,
                shuffle=False,
            ),
        ):
            model = RBuildAdaptedModel(ToyCausalLM(), tiny_v41(memory=False, critics=False))
            before = model.base_model.head.weight.detach().clone()
            trainer = RBuildBaseTrainer(model, model.config, device="cpu", log_fn=None)
            result = trainer.fit(batches, max_steps=1)
            self.assertEqual(result["steps"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
            self.assertFalse(torch.equal(before, model.base_model.head.weight))

    def test_release_gate_rejects_capability_regression(self):
        base = ToyCausalLM().eval()
        output = base(self.ids)
        self.assertFalse(v41_release_gate(output, output, 1.0, 0.9)["passed"])
        other = SimpleNamespace(logits=output.logits + 1)
        self.assertFalse(v41_release_gate(output, other)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
