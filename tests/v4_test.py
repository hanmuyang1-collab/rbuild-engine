"""Focused R-Build-v4 acceptance tests (standard-library unittest)."""
import unittest

import torch

from rbuild import RBuildConfig, RBuildModel, ROutGenModel, count_routgen_parameters
from rbuild.v4 import v4_release_gate


def tiny_config():
    cfg = RBuildConfig()
    cfg.model.max_seq_len = 64
    cfg.critic.min_loops = 1
    cfg.critic.max_loops = 1
    cfg.noting.enabled = False
    cfg.actuation.enabled = False
    cfg.memory.enabled = False
    cfg.draft_verify.n_layers = 1
    cfg.draft_verify.d_model = 32
    cfg.draft_verify.ffn_dim = 64
    cfg.draft_verify.n_heads = 4
    cfg.draft_verify.draft_len = 4
    cfg.draft_verify.denoise_steps = 2
    cfg.draft_verify.max_rounds = 2
    cfg.outgen.enabled = False
    cfg.validate()
    return cfg


class V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.cfg = tiny_config()
        cls.model = RBuildModel(cls.cfg).eval()
        cls.prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def test_counter_and_shared_embeddings(self):
        actual = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(actual, self.cfg.count_parameters()["total_params"])
        self.assertIsNotNone(self.model.draft)
        self.assertEqual(self.model.draft.blocks.__len__(), 1)

    def test_training_loss_and_kl(self):
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        hidden, _, _ = self.model._encode(x, stable_inference=True)
        teacher = self.model._logits_from_hidden(hidden)
        ce, kl = self.model.draft.training_loss(
            x, y, teacher, self.model.embed, self.model.lm_head.weight)
        self.assertTrue(torch.isfinite(ce))
        self.assertTrue(torch.isfinite(kl))
        (ce + 0.05 * kl).backward()
        grads = [p.grad for p in self.model.draft.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def assert_ar_equal(self, watermark, wrong_draft=False):
        draft = self.model.draft
        original = draft.propose
        if wrong_draft:
            def reject_all(*args, **kwargs):
                proposed = original(*args, **kwargs)
                return (proposed + 1) % self.model.embed.num_embeddings
            draft.propose = reject_all
        try:
            torch.manual_seed(1234)
            ar = self.model.generate(self.prompt, max_new_tokens=7,
                                     watermark=watermark, use_v4=False)
            torch.manual_seed(1234)
            accelerated = self.model.generate(self.prompt, max_new_tokens=7,
                                              watermark=watermark, use_v4=True)
            self.assertTrue(torch.equal(ar, accelerated))
            self.assertEqual(ar.numpy().tobytes(), accelerated.numpy().tobytes())
            if wrong_draft:
                self.assertEqual(self.model.last_v4_stats["accepted"], 0)
                self.assertGreater(self.model.last_v4_stats["proposed"], 0)
        finally:
            draft.propose = original

    def test_exact_ar_equivalence(self):
        self.assert_ar_equal(watermark=False)

    def test_exact_ar_equivalence_with_watermark(self):
        self.assert_ar_equal(watermark=True)

    def test_rejected_draft_tokens_never_escape(self):
        self.assert_ar_equal(watermark=True, wrong_draft=True)

    def test_release_gates_and_freeze(self):
        good = v4_release_gate(0.60, b"abc", b"abc", 1.0, 1.0)
        self.assertTrue(good["passed"])
        self.assertFalse(v4_release_gate(0.59, b"abc", b"abc", 1.0, 1.0)["passed"])
        self.assertFalse(v4_release_gate(0.9, b"abc", b"abd", 1.0, 1.0)["passed"])
        self.assertFalse(v4_release_gate(0.9, b"abc", b"abc", 1.0, 0.9)["passed"])
        with self.assertRaises(ValueError):
            self.model.freeze_v4({"passed": False})
        self.model.freeze_v4(good)
        self.assertTrue(all(not p.requires_grad for p in self.model.draft.parameters()))
        for p in self.model.draft.parameters():
            p.requires_grad_(True)
        self.cfg.draft_verify.frozen = False

    def test_routgen_excludes_text_draft_counter(self):
        cfg = tiny_config()
        cfg.outgen.enabled = True
        cfg.outgen.image = True
        cfg.outgen.video = cfg.outgen.tts = False
        cfg.outgen.image_token_id = 250
        model = ROutGenModel(cfg)
        actual = sum(p.numel() for p in model.parameters())
        self.assertEqual(actual, count_routgen_parameters(cfg)["total_params"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
