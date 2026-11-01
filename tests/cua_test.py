"""Universal CUA acceptance tests for native, media, and adapted models."""

import os
import tempfile
import unittest

import torch

from rbuild import (
    Action,
    ConfirmationRequired,
    RBuildAdaptedModel,
    RBuildConfig,
    RBuildModel,
    ROutGenModel,
    Trainer,
    count_routgen_parameters,
)
from tests.v41_test import ToyCausalLM, tiny_v41


def tiny_native(outgen=False):
    cfg = RBuildConfig()
    cfg.model.vocab_size = 64
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 4
    cfg.model.n_layers = 1
    cfg.model.max_seq_len = 32
    cfg.memory.key_dim = 16
    cfg.memory.value_dim = 16
    cfg.memory.enabled = False
    cfg.critic.n_critics = 2
    cfg.critic.y_critics = 1
    cfg.critic.max_loops = 1
    cfg.parallel.n_stages = 1
    cfg.noting.enabled = False
    cfg.draft_verify.enabled = False
    cfg.actuation.screen_grid = 4
    cfg.actuation.scroll_steps = 2
    cfg.outgen.enabled = outgen
    cfg.outgen.image = outgen
    cfg.outgen.video = False
    cfg.outgen.tts = False
    cfg.outgen.image_token_id = 10
    cfg.outgen.image_size = 8
    cfg.outgen.image_patch = 4
    cfg.validate()
    return cfg


class FakeDriver:
    def __init__(self):
        self.events = []

    def moveTo(self, x, y):
        self.events.append(("move", x, y))

    def click(self):
        self.events.append(("click",))

    def scroll(self, dy):
        self.events.append(("scroll", dy))

    def write(self, text):
        self.events.append(("type", text))


class UniversalCUATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_native_cua_and_legacy_logit_compatibility(self):
        for merged in (True, False):
            cfg = tiny_native()
            cfg.actuation.enabled = merged
            native = RBuildModel(cfg).eval()
            ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            logits, _ = native(ids)
            expected_vocab = cfg.effective_vocab_size() if merged else cfg.model.vocab_size
            self.assertEqual(logits.shape[-1], expected_vocab)
            decision = native.cua.observe(ids)
            self.assertEqual(len(decision.actions), 1)
            hidden = native.hidden_states(ids)
            labels = torch.full(ids.shape, native.action_codec.wait_id)
            loss = native.cua.action_loss(hidden, labels)
            loss.backward()
            self.assertIsNotNone(native.action_head.proj.weight.grad)
            self.assertEqual(
                sum(p.numel() for p in native.parameters()),
                cfg.count_parameters()["total_params"],
            )

    def test_routgen_has_same_cua_contract(self):
        cfg = tiny_native(outgen=True)
        model = ROutGenModel(cfg).eval()
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        decision = model.cua.observe(ids)
        self.assertEqual(len(decision.actions), 1)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()),
            count_routgen_parameters(cfg)["total_params"],
        )
        self.assertEqual(model.action_codec.n_actions, cfg.n_action_tokens())

    def test_adapted_cua_checkpoint_roundtrip(self):
        model = RBuildAdaptedModel(ToyCausalLM(), tiny_v41())
        hidden = torch.randn(1, 3, 16)
        decision = model.cua.propose(hidden)
        self.assertEqual(len(decision.actions), 1)
        loss = model.cua.action_loss(
            hidden, torch.full((1, 3), model.action_codec.wait_id)
        )
        self.assertTrue(torch.isfinite(loss))
        with tempfile.TemporaryDirectory() as path:
            model.save_checkpoint(path)
            restored = RBuildAdaptedModel.load_checkpoint(path, base_model=ToyCausalLM())
            self.assertTrue(torch.equal(
                model.action_head.proj.weight, restored.action_head.proj.weight
            ))
            self.assertEqual(restored.cua.propose(hidden).token_ids.tolist(), decision.token_ids.tolist())

    def test_pre_cua_native_checkpoints_remain_loadable(self):
        for outgen in (False, True):
            cfg = tiny_native(outgen=outgen)
            model = ROutGenModel(cfg) if outgen else RBuildModel(cfg)
            state = model.state_dict()
            state.pop("action_head.proj.weight")
            with tempfile.TemporaryDirectory() as path:
                cfg.save(os.path.join(path, "rbuild_config.json"))
                torch.save(state, os.path.join(path, "model.pt"))
                restored = (
                    ROutGenModel.load_checkpoint(path)
                    if outgen else Trainer.load_checkpoint(path)
                )
                self.assertTrue(restored.cua.restored_from_legacy)
                self.assertIsNotNone(restored.action_head)

    def test_codec_and_driver_execution(self):
        model = RBuildAdaptedModel(ToyCausalLM(), tiny_v41())
        click_id = model.action_codec.encode_click(0.25, 0.75)
        scroll_id = model.action_codec.encode_scroll(-2)
        decoded = model.cua.decode([click_id, scroll_id])
        self.assertEqual([action.kind for action in decoded], ["click", "scroll"])
        driver = FakeDriver()
        model.cua.execute(decoded + [Action("type", text="hello")], driver)
        self.assertEqual(driver.events[-1], ("type", "hello"))
        self.assertEqual(driver.events[1], ("click",))

    def test_irreversible_guard_blocks_before_any_side_effect(self):
        model = RBuildAdaptedModel(ToyCausalLM(), tiny_v41())
        driver = FakeDriver()
        actions = [
            Action("click", x=0.5, y=0.5),
            Action("type", text="send", irreversible=True),
        ]
        with self.assertRaises(ConfirmationRequired):
            model.cua.execute(actions, driver)
        self.assertEqual(driver.events, [])
        model.cua.execute(actions, driver, confirmed=True)
        self.assertEqual(driver.events[-1], ("type", "send"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
