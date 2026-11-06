"""Acceptance tests for the complete CUA expansion on the original engine."""

import os
import tempfile
import unittest

import torch

from rbuild import (
    Action,
    ConfirmationRequired,
    CUAReplayLedger,
    CUAPolicy,
    CUATrajectory,
    CUATrajectoryBuffer,
    PolicyDriver,
    PolicyViolation,
    PostconditionFailed,
    RBuildAdaptedModel,
    RBuildBaseTrainer,
    RBuildModel,
    ROutGenModel,
    RECOVERY_EVENTS,
    ScreenChangeDetector,
    ShadowDriver,
    build_recovery_curriculum,
    count_routgen_parameters,
    cua_release_gate,
    run_cua_contract_canaries,
)
from tests.cua_test import FakeDriver, tiny_native
from tests.v41_test import ToyCausalLM, tiny_v41


class CUAExpansionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def adapted(self, **config_changes):
        config = tiny_v41(memory=False, critics=False)
        for name, value in config_changes.items():
            setattr(config, name, value)
        return RBuildAdaptedModel(ToyCausalLM(), config)

    def test_uncertainty_abstention_and_confirmation(self):
        model = self.adapted()
        hidden = torch.randn(2, 3, 16)
        decision = model.cua.propose(hidden, max_entropy=-1, on_uncertain="wait")
        self.assertTrue(bool(decision.abstained.all()))
        self.assertTrue(bool((decision.token_ids == model.action_codec.wait_id).all()))
        self.assertEqual([action.kind for action in decision.actions], ["wait", "wait"])
        confirm = model.cua.propose(hidden, max_entropy=-1, on_uncertain="confirm")
        self.assertTrue(bool(confirm.needs_confirmation.all()))
        driver = FakeDriver()
        with self.assertRaises(ConfirmationRequired):
            model.cua.act(hidden, driver, max_entropy=-1, on_uncertain="confirm")
        self.assertEqual(driver.events, [])

    def test_critic_candidates_and_postconditions(self):
        config = tiny_v41(memory=False, critics=True)
        model = RBuildAdaptedModel(ToyCausalLM(), config)
        hidden = torch.randn(1, 2, 16)
        decision = model.cua.propose_verified(
            hidden, candidates=3, min_votes=0, min_satisfaction=-1
        )
        self.assertEqual(decision.verification["candidate_votes"].shape, (1, 3))
        self.assertFalse(bool(decision.abstained.any()))
        rejected = model.cua.propose_verified(hidden, candidates=3, min_votes=99)
        self.assertTrue(bool(rejected.abstained.all()))
        self.assertEqual(int(rejected.token_ids[0]), model.action_codec.wait_id)
        driver = ShadowDriver()
        driver.postconditions["ready"] = True
        model.cua.act_verified(
            hidden,
            driver,
            candidates=2,
            min_votes=0,
            expected_postcondition="ready",
        )
        failed = ShadowDriver()
        with self.assertRaises(PostconditionFailed):
            model.cua.execute(
                [Action("click", x=0.5, y=0.5)], failed, postcondition="missing"
            )
        self.assertEqual(model.cua.ledger.records[-1]["status"], "failed")

    def test_event_driven_observation(self):
        model = RBuildModel(tiny_native()).eval()
        ids = torch.tensor([[1, 2, 3]])
        detector = ScreenChangeDetector(threshold=0.1)
        screen = torch.zeros(3, 8, 8)
        first = model.cua.observe_if_changed(screen, ids, detector=detector)
        second = model.cua.observe_if_changed(screen, ids, detector=detector)
        changed = model.cua.observe_if_changed(torch.ones(3, 8, 8), ids, detector=detector)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(changed)
        timed = ScreenChangeDetector(threshold=1.0, min_interval=2.0, max_interval=5.0)
        self.assertTrue(timed.should_observe(screen, now=0.0))
        self.assertFalse(timed.should_observe(torch.ones(3, 8, 8), now=1.0))
        self.assertTrue(timed.should_observe(torch.ones(3, 8, 8), now=5.0))

    def test_shadow_mode_and_application_policies(self):
        model = self.adapted()
        real = FakeDriver()
        shadow = model.cua.shadow(real)
        model.cua.execute([Action("click", x=0.5, y=0.5), Action("type", text="safe")], shadow)
        self.assertEqual(real.events, [])
        self.assertEqual(len(shadow.events), 3)
        policy = CUAPolicy(
            allowed_kinds=("click", "type"),
            confirmation_kinds=("click",),
            blocked_text=("delete",),
            min_interval_seconds=100.0,
        )
        driver = PolicyDriver(ShadowDriver(), policy)
        with self.assertRaises(PolicyViolation):
            model.cua.execute([Action("click", x=2.0, y=0.5)], driver)
        with self.assertRaises(PolicyViolation):
            model.cua.execute([Action("type", text="delete all")], driver)
        with self.assertRaises(ConfirmationRequired):
            model.cua.execute([Action("click", x=0.5, y=0.5)], driver)
        model.cua.execute([Action("click", x=0.5, y=0.5)], driver, confirmed=True)
        with self.assertRaises(PolicyViolation):
            model.cua.execute([Action("type", text="safe")], driver)

    def test_audit_persistence_and_deterministic_replay(self):
        model = self.adapted()
        with tempfile.TemporaryDirectory() as path:
            audit = os.path.join(path, "cua.jsonl")
            ledger = CUAReplayLedger(path=audit)
            original = FakeDriver()
            actions = [Action("click", x=0.25, y=0.75), Action("type", text="hello")]
            model.cua.execute(actions, original, ledger=ledger, metadata={"case": "replay"})
            self.assertTrue(os.path.exists(audit))
            restored = CUAReplayLedger()
            restored.load_jsonl(audit)
            replayed = FakeDriver()
            restored.replay(model.cua, 0, replayed)
            self.assertEqual(replayed.events, original.events)
            self.assertEqual(restored.records[0]["metadata"]["case"], "replay")

    def test_verified_replay_and_recovery_curriculum(self):
        model = self.adapted()
        codec = model.action_codec
        labels = torch.tensor([[codec.encode_click(0.5, 0.5), codec.wait_id]])
        trajectory = CUATrajectory(torch.randn(1, 2, 16), labels, verified=True)
        buffer = CUATrajectoryBuffer(max_trajectories=16)
        with self.assertRaises(ValueError):
            buffer.add(CUATrajectory(torch.randn(1, 2, 16), labels, verified=False))
        buffer.add(trajectory)
        buffer.add(CUATrajectory(torch.randn(1, 1, 16), labels[:, :1], verified=True))
        batch = next(buffer.batches(batch_size=2, shuffle=False))
        self.assertEqual(int((batch["action_token_ids"] == -100).sum()), 1)
        self.assertTrue(torch.isfinite(model.cua.action_loss(
            batch["hidden_states"], batch["action_token_ids"]
        )))
        self.assertEqual(buffer.items[0].hidden_states.dtype, torch.float16)
        variants = build_recovery_curriculum(codec, [trajectory])
        self.assertEqual({item.metadata["recovery_event"] for item in variants}, set(RECOVERY_EVENTS))
        self.assertTrue(all(item.verified for item in variants))

    def test_contract_canaries_and_release_gate(self):
        models = {
            "native": RBuildModel(tiny_native()).eval(),
            "routgen": ROutGenModel(tiny_native(outgen=True)).eval(),
            "adapted": self.adapted(),
        }
        canaries = run_cua_contract_canaries(models)
        self.assertTrue(all(canaries.values()), canaries)
        self.assertTrue(cua_release_gate(canaries, required=list(canaries))["passed"])
        canaries["irreversible_action_rejection"] = False
        gate = cua_release_gate(canaries, required=list(canaries))
        self.assertFalse(gate["passed"])
        self.assertIn("irreversible_action_rejection", gate["safety_failed"])

    def test_frozen_base_cua_training(self):
        model = self.adapted()
        model.freeze_for_cua()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base_model.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.action_head.parameters()))
        base_before = model.base_model.head.weight.detach().clone()
        action_before = model.action_head.proj.weight.detach().clone()
        trainer = RBuildBaseTrainer(model, model.config, device="cpu", log_fn=None)
        buffer = CUATrajectoryBuffer()
        buffer.add(CUATrajectory(
            torch.randn(1, 3, 16),
            torch.full((1, 3), model.action_codec.wait_id),
            verified=True,
        ))
        result = trainer.fit_cua(buffer.batches(shuffle=False), max_steps=1)
        self.assertEqual(result["steps"], 1)
        self.assertTrue(torch.equal(base_before, model.base_model.head.weight))
        self.assertFalse(torch.equal(action_before, model.action_head.proj.weight))

    def test_encoderless_screenshots_preserve_base_outputs(self):
        model = self.adapted(
            cua_screenshot_enabled=True,
            cua_screenshot_size=8,
            cua_screenshot_patch=4,
        )
        ids = torch.tensor([[2, 3, 4, 5]])
        expected = model.base_model(ids)
        actual = model(ids)
        self.assertTrue(torch.equal(expected.logits, actual.logits))
        screenshots = torch.randn(2, 3, 8, 8)
        hidden = model.screenshot_hidden(screenshots)
        self.assertEqual(hidden.shape, (2, 4, 16))
        decision = model.cua.observe_screenshot(screenshots)
        self.assertEqual(decision.token_ids.shape, (2,))
        projector_params = sum(p.numel() for p in model.screenshot_projector.parameters())
        self.assertEqual(projector_params, 3 * 4 * 4 * 16)

    def test_application_adapters_and_checkpoint_roundtrip(self):
        model = self.adapted()
        model.add_cua_adapter("desktop", rank=2, alpha=4.0)
        model.activate_cua_adapter("desktop")
        hidden = torch.randn(1, 3, 16)
        self.assertTrue(torch.equal(hidden, model.transform_cua_hidden(hidden)))
        with torch.no_grad():
            model.cua_adapters["desktop"].up.weight.fill_(0.01)
        self.assertFalse(torch.equal(hidden, model.transform_cua_hidden(hidden)))
        expected = model.cua.propose(hidden).token_ids
        with tempfile.TemporaryDirectory() as path:
            model.save_checkpoint(path)
            restored = RBuildAdaptedModel.load_checkpoint(path, base_model=ToyCausalLM())
            self.assertEqual(restored._active_cua_adapter, "desktop")
            self.assertEqual(restored.cua_adapters["desktop"].rank, 2)
            self.assertTrue(torch.equal(
                model.cua_adapters["desktop"].up.weight,
                restored.cua_adapters["desktop"].up.weight,
            ))
            self.assertTrue(torch.equal(expected, restored.cua.propose(hidden).token_ids))

    def test_runtime_expansion_adds_no_native_parameters(self):
        native_cfg = tiny_native()
        native = RBuildModel(native_cfg)
        routgen_cfg = tiny_native(outgen=True)
        routgen = ROutGenModel(routgen_cfg)
        ScreenChangeDetector()
        CUAReplayLedger()
        CUAPolicy()
        CUATrajectoryBuffer()
        self.assertEqual(
            sum(p.numel() for p in native.parameters()),
            native_cfg.count_parameters()["total_params"],
        )
        self.assertEqual(
            sum(p.numel() for p in routgen.parameters()),
            count_routgen_parameters(routgen_cfg)["total_params"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
