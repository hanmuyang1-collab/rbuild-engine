"""R-Build v4.2 acceptance tests (standard-library unittest)."""

import copy
import os
import tempfile
import unittest
from unittest import mock

import torch

from rbuild import (
    RBuildConfig,
    RBuildModel,
    StagePLedger,
    Trainer,
    V42DataSource,
    V42DistributedConfig,
    V42Ladder,
    V42ScratchTrainer,
    V42Stage,
    V42StreamingData,
    V42ThroughputProfile,
    default_v42_data_sources,
    estimate_stage_p_cost,
    v42_release_gate,
)


def tiny_config(max_steps=2):
    cfg = RBuildConfig()
    cfg.model.vocab_size = 64
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.max_seq_len = 16
    cfg.cache_loop.n_layers = 1
    cfg.cache_loop.n_loops = 1
    cfg.cache_loop.use_mod_routing = True
    cfg.critic.enabled = False
    cfg.noting.enabled = False
    cfg.parallel.n_stages = 1
    cfg.parallel.n_branches = 1
    cfg.parallel.n_experts = 2
    cfg.parallel.expert_top_k = 1
    cfg.parallel.expert_ffn_dim = 16
    cfg.parallel.use_mod_routing = True
    cfg.memory.key_dim = 8
    cfg.memory.value_dim = 32
    cfg.draft_verify.enabled = False
    cfg.actuation.screen_grid = 4
    cfg.train.batch_size = 1
    cfg.train.grad_accum = 1
    cfg.train.max_steps = max_steps
    cfg.train.warmup_steps = 1
    cfg.train.precision = "fp32"
    cfg.train.lr = 1e-3
    cfg.train.adamw_lr = 1e-3
    return cfg


def clone_config(cfg):
    return RBuildConfig.from_dict(copy.deepcopy(cfg.to_dict()))


def tiny_ladder(cfg=None, production=False):
    cfg = cfg or tiny_config()
    stages = [V42Stage(name, clone_config(cfg), max_steps=cfg.train.max_steps)
              for name in ("A", "B", "C", "D")]
    if production:
        final = V42Stage("P", clone_config(cfg), max_steps=cfg.train.max_steps)
        return V42Ladder(stages + [final], production=True)
    return V42Ladder(stages + [V42Stage("E", clone_config(cfg),
                                       max_steps=cfg.train.max_steps)])


def batches():
    x = torch.arange(16, dtype=torch.long).remainder(64).unsqueeze(0)
    y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
    return [(x, y)]


class V42LedgerTests(unittest.TestCase):
    def test_locked_stage_p_math(self):
        ledger = StagePLedger()
        ledger.validate()
        self.assertEqual(ledger.total_params, 820_000_000_000)
        self.assertEqual(ledger.active_expert_params, 82_000_000_000)
        self.assertEqual(ledger.active_expert_params * 10, ledger.total_params)
        ledger.params_per_expert = 51_250_000_000
        with self.assertRaises(ValueError):
            ledger.validate()

    def test_h200_layout_and_memory(self):
        distributed = V42DistributedConfig()
        distributed.validate()
        self.assertEqual(distributed.gpu_count, 128)
        placements = distributed.expert_placements()
        self.assertEqual(len(placements), 16)
        self.assertTrue(all(len(items) == 2 for items in placements.values()))
        for expert in range(16):
            roles = [item["role"] for items in placements.values()
                     for item in items if item["expert"] == expert]
            self.assertEqual(sorted(roles), ["primary", "replica"])
        memory = distributed.memory_ledger()
        self.assertEqual(memory.aggregate_vram_gb, 18_048)
        self.assertEqual(memory.training_resident_gb, 13_120)
        self.assertEqual(memory.remaining_activation_budget_gb, 4_928)
        self.assertEqual(memory.training_resident_per_gpu_gb, 102.5)
        command = distributed.torchrun_command(
            "examples/train_scratch_v42.py", 3, "10.0.0.1", ["--dry-run"]
        )
        self.assertIn("--node-rank=3", command)
        self.assertIn("--nproc-per-node=8", command)
        self.assertEqual(command[-1], "--dry-run")

    def test_10m_context_and_ep_guards(self):
        distributed = V42DistributedConfig(full_activation_recompute=False)
        with self.assertRaises(ValueError):
            distributed.validate()
        distributed = V42DistributedConfig(expert_parallel=16)
        with self.assertRaises(ValueError):
            distributed.validate()

    def test_two_column_cost_requires_explicit_assumptions(self):
        result = estimate_stage_p_cost(
            1_000_000_000,
            V42ThroughputProfile(2.0, 100.0, efficiency=0.5),
            V42ThroughputProfile(2.0, 400.0, efficiency=0.75),
        )
        self.assertIn("naive", result)
        self.assertIn("optimized", result)
        self.assertGreater(result["naive"]["cost_usd"], result["optimized"]["cost_usd"])
        self.assertIn("speedup_x", result)


class V42LadderTests(unittest.TestCase):
    def test_open_intermediate_sizes_survive_json(self):
        ladder = V42Ladder.production_ladder([V42Stage(str(i)) for i in range(4)])
        ladder.validate()
        self.assertIsNone(ladder.stages[0].config)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "ladder.json")
            ladder.save(path)
            restored = V42Ladder.load(path)
        self.assertTrue(restored.production)
        self.assertEqual(restored.stages[-1].total_params, 820_000_000_000)
        with self.assertRaises(ValueError):
            restored.validate(allow_open=False)

    def test_native_counter_still_matches_actual_model(self):
        cfg = tiny_config()
        model = RBuildModel(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         cfg.count_parameters()["total_params"])

    def test_production_stage_p_config_cannot_contradict_ledger(self):
        ladder = tiny_ladder(production=True)
        with self.assertRaisesRegex(ValueError, "820B"):
            ladder.validate()

    def test_release_gate_reuses_v41_preservation(self):
        output = torch.tensor([[1.0, 2.0]])
        ladder = tiny_ladder()
        gate = v42_release_gate(output, output, 1.0, 1.0,
                                ladder=ladder, distributed=V42DistributedConfig())
        self.assertTrue(gate["passed"])
        self.assertFalse(v42_release_gate(output, output, 1.0, 0.9,
                                          ladder=ladder)["passed"])
        self.assertFalse(v42_release_gate(output, output + 1,
                                          ladder=ladder)["passed"])


class V42DataTests(unittest.TestCase):
    def test_default_mix_has_exactly_one_math_source(self):
        sources = default_v42_data_sources("megamath")
        self.assertEqual(len(sources), 4)
        self.assertEqual(sum(source.name == "math" for source in sources), 1)
        self.assertEqual(sources[2].dataset, "LLM360/MegaMath")
        with self.assertRaises(ValueError):
            default_v42_data_sources("both")

    def test_admission_filter_runs_before_batches(self):
        source = V42DataSource(
            "local", "local/test", license="approved-test", license_approved=True,
            pii_policy="reject-test-secrets",
            admission_filter=lambda row: "secret" not in row["text"],
        )
        stream = V42StreamingData(
            [source], production=True,
            loaders={"local": lambda: iter([
                {"text": "keep one"}, {"text": "secret value"}, {"text": "keep two"},
            ])},
        )
        self.assertEqual(list(stream.rows()), ["keep one", "keep two"])
        cfg = tiny_config()
        batch = next(iter(stream.batches(cfg)))
        self.assertEqual(batch[0].shape, (1, 16))
        self.assertEqual(batch[1].shape, (1, 16))

    def test_production_data_requires_license_and_filter(self):
        with self.assertRaises(ValueError):
            V42StreamingData([V42DataSource("code", "bigcode/the-stack-v2")],
                             production=True)


class V42ScratchTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_from_scratch_strict_growth_and_exact_resume(self):
        ladder = tiny_ladder(tiny_config(max_steps=4))
        with tempfile.TemporaryDirectory() as root:
            runner = V42ScratchTrainer(ladder, device="cpu", log_fn=None)
            first = runner.run_stage(0, batches(), root, stop_step=2)
            self.assertEqual(first["metadata"]["weight_transfer"]["mode"], "from-scratch")
            self.assertEqual(first["training"]["steps"], 2)
            self.assertTrue(os.path.exists(os.path.join(
                first["checkpoint"], "v42_trainer.pt")))

            second = runner.run_stage(1, batches(), root)
            self.assertEqual(second["metadata"]["weight_transfer"]["mode"],
                             "strict-identical-geometry")

            resumed = runner.run_stage(0, batches(), root,
                                       resume_checkpoint=first["checkpoint"])
            self.assertEqual(resumed["training"]["steps"], 4)
            self.assertEqual(resumed["metadata"]["weight_transfer"]["mode"], "exact-resume")

            uninterrupted = runner.run_stage(0, batches(), os.path.join(root, "full"))
            resumed_model = Trainer.load_checkpoint(resumed["checkpoint"])
            full_model = Trainer.load_checkpoint(uninterrupted["checkpoint"])
            for key, value in resumed_model.state_dict().items():
                self.assertTrue(torch.equal(value, full_model.state_dict()[key]), key)

    def test_shape_change_refuses_silent_weight_reset(self):
        ladder = tiny_ladder()
        changed = clone_config(ladder.stages[0].config)
        changed.model.d_model = 64
        changed.memory.value_dim = 64
        ladder.stages[1].config = changed
        with tempfile.TemporaryDirectory() as root:
            runner = V42ScratchTrainer(ladder, device="cpu", log_fn=None)
            runner.run_stage(0, batches(), root)
            with self.assertRaisesRegex(ValueError, "growth_fn"):
                runner.run_stage(1, batches(), root)

    def test_production_stage_p_requires_sharding_and_release_evidence(self):
        ladder = tiny_ladder(production=True)
        # Isolate the runtime guard; the ledger/config contradiction is tested above.
        with mock.patch.object(V42Ladder, "validate", lambda self, allow_open=True: None):
            runner = V42ScratchTrainer(ladder, device="cpu", log_fn=None)
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(RuntimeError, "sharding wrap_fn"):
                    runner.run_stage(4, batches(), root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
