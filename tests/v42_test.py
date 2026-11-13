"""R-Build v4.2 acceptance tests (standard-library unittest)."""

import copy
import os
import tempfile
import unittest

import torch

import rbuild
from rbuild import (
    RBuildConfig,
    RBuildModel,
    Trainer,
    V42CostTarget,
    V42DataSource,
    V42DistributedConfig,
    V42Ladder,
    V42ScratchTrainer,
    V42Stage,
    V42StreamingData,
    V42ThroughputProfile,
    estimate_v42_cost,
    optimize_v42_cost,
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
    return V42Ladder([
        V42Stage(name, clone_config(cfg), max_steps=cfg.train.max_steps)
        for name in ("custom-a", "custom-b")
    ], production=production)


def custom_distributed(**overrides):
    values = dict(
        nodes=2,
        gpus_per_node=4,
        gpu_vram_gb=80,
        tensor_parallel=2,
        n_experts=4,
        expert_parallel=8,
        expert_replicas=2,
        expert_placements_per_node=4,
        max_context=1024,
        interconnect="test-fabric",
        intranode_interconnect="test-link",
    )
    values.update(overrides)
    return V42DistributedConfig(**values)


def batches():
    x = torch.arange(16, dtype=torch.long).remainder(64).unsqueeze(0)
    y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
    return [(x, y)]


class V42CallerConfigurationTests(unittest.TestCase):
    def test_custom_distributed_layout_memory_and_json(self):
        distributed = custom_distributed()
        distributed.validate()
        self.assertEqual(distributed.gpu_count, 8)
        placements = distributed.expert_placements()
        self.assertEqual(len(placements), 2)
        self.assertTrue(all(len(items) == 4 for items in placements.values()))
        for expert in range(4):
            roles = [item["role"] for items in placements.values()
                     for item in items if item["expert"] == expert]
            self.assertEqual(sorted(roles), ["primary", "replica"])
        memory = distributed.memory_ledger(1_000_000_000)
        self.assertEqual(memory.aggregate_vram_gb, 640)
        self.assertEqual(memory.training_resident_gb, 16)
        self.assertEqual(memory.remaining_activation_budget_gb, 624)
        self.assertEqual(memory.training_resident_per_gpu_gb, 2)
        command = distributed.torchrun_command(
            "examples/train_scratch_v42.py", 1, "10.0.0.1", ["--dry-run"]
        )
        self.assertIn("--node-rank=1", command)
        self.assertIn("--nproc-per-node=4", command)
        self.assertEqual(command[-1], "--dry-run")
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "plan.json")
            distributed.save(path)
            self.assertEqual(V42DistributedConfig.load(path).to_dict(),
                             distributed.to_dict())

    def test_context_and_expert_guards_use_caller_dimensions(self):
        with self.assertRaises(ValueError):
            custom_distributed(max_context=10_000_000,
                               full_activation_recompute=False).validate()
        with self.assertRaises(ValueError):
            custom_distributed(expert_parallel=4).validate()
        with self.assertRaises(ValueError):
            custom_distributed(expert_placements_per_node=3).validate()

    def test_two_column_cost_uses_explicit_sizes_and_hardware(self):
        result = estimate_v42_cost(
            total_tokens=1_000_000_000,
            total_params=1_000_000,
            active_params=200_000,
            naive=V42ThroughputProfile(2.0, 100.0, efficiency=0.5),
            optimized=V42ThroughputProfile(2.0, 400.0, efficiency=0.75),
            distributed=custom_distributed(),
        )
        self.assertIn("naive", result)
        self.assertIn("optimized", result)
        self.assertEqual(result["naive"]["flops_per_token"], 6_000_000)
        self.assertEqual(result["optimized"]["flops_per_token"], 1_200_000)
        self.assertGreater(result["naive"]["cost_usd"], result["optimized"]["cost_usd"])
        self.assertIn("speedup_x", result)

    def test_default_target_derives_300k_to_120k_profile(self):
        inputs = dict(
            total_tokens=27_000_000_000_000_000_000_000,
            total_params=1_000_000,
            active_params=200_000,
            naive=V42ThroughputProfile(2.0, 100.0, efficiency=0.5,
                                       flops_per_token=1.0),
            optimized=V42ThroughputProfile(2.0, 100.0, efficiency=1.0,
                                           flops_per_token=1.0),
            distributed=custom_distributed(),
        )
        result = optimize_v42_cost(**inputs)
        self.assertAlmostEqual(result["naive"]["cost_usd"], 300_000.0)
        self.assertAlmostEqual(result["optimized"]["cost_usd"], 150_000.0)
        self.assertTrue(result["cost_target"]["naive_within_reference_band"])
        self.assertFalse(result["cost_target"]["current_target_met"])
        self.assertEqual(result["cost_target"]["reference_savings_usd"], 180_000.0)
        self.assertEqual(result["cost_target"]["reference_reduction_pct"], 60.0)
        self.assertAlmostEqual(result["cost_target"]["additional_speedup_x"], 1.25)
        self.assertAlmostEqual(result["required_speedup_vs_naive_x"], 2.5)

        required = result["required_optimized_profile"]
        self.assertAlmostEqual(required["achieved_tflops_per_gpu"], 125.0)
        self.assertAlmostEqual(required["projected_cost_usd"], 120_000.0)
        resolved_inputs = dict(inputs)
        resolved_inputs["optimized"] = V42ThroughputProfile(
            required["gpu_price_per_hour"],
            required["achieved_tflops_per_gpu"],
            efficiency=required["efficiency"],
            flops_per_token=required["flops_per_token"],
        )
        resolved = estimate_v42_cost(**resolved_inputs)
        self.assertAlmostEqual(resolved["optimized"]["cost_usd"], 120_000.0)

    def test_cost_target_never_downgrades_an_already_meeting_profile(self):
        result = optimize_v42_cost(
            total_tokens=27_000_000_000_000_000_000_000,
            total_params=1_000_000,
            active_params=200_000,
            naive=V42ThroughputProfile(2.0, 100.0, 0.5, flops_per_token=1.0),
            optimized=V42ThroughputProfile(2.0, 200.0, 1.0, flops_per_token=1.0),
            distributed=custom_distributed(),
        )
        self.assertTrue(result["cost_target"]["current_target_met"])
        self.assertEqual(result["cost_target"]["additional_speedup_x"], 1.0)
        self.assertEqual(
            result["required_optimized_profile"]["achieved_tflops_per_gpu"], 200.0
        )
        self.assertAlmostEqual(
            result["required_optimized_profile"]["projected_cost_usd"], 75_000.0
        )

    def test_cost_target_is_configurable_and_rejects_invalid_budgets(self):
        target = V42CostTarget(baseline_usd=200_000.0, target_usd=80_000.0,
                               tolerance=0.1)
        self.assertEqual(target.to_dict()["reference_reduction_pct"], 60.0)
        invalid = (
            V42CostTarget(baseline_usd=0.0),
            V42CostTarget(target_usd=float("nan")),
            V42CostTarget(tolerance=float("inf")),
            V42CostTarget(baseline_usd=100_000.0, target_usd=120_000.0),
            V42CostTarget(tolerance=1.0),
        )
        for bad_target in invalid:
            with self.subTest(target=bad_target), self.assertRaises(ValueError):
                bad_target.validate()


class V42LadderTests(unittest.TestCase):
    def test_open_caller_ladder_survives_json_without_fixed_stage_count(self):
        ladder = V42Ladder([V42Stage("only-stage", total_params=123, active_params=45)])
        ladder.validate()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "ladder.json")
            ladder.save(path)
            restored = V42Ladder.load(path)
        self.assertEqual(len(restored.stages), 1)
        self.assertEqual(restored.stages[0].name, "only-stage")
        self.assertEqual(restored.stages[0].total_params, 123)
        self.assertEqual(restored.stages[0].active_params, 45)
        with self.assertRaises(ValueError):
            restored.validate(allow_open=False)

    def test_native_counter_still_matches_actual_model(self):
        cfg = tiny_config()
        model = RBuildModel(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         cfg.count_parameters()["total_params"])

    def test_custom_production_weights_are_accepted_but_declarations_must_match(self):
        ladder = tiny_ladder(production=True)
        ladder.validate(allow_open=False)
        ladder.stages[-1].total_params = (
            ladder.stages[-1].config.count_parameters()["total_params"] + 1
        )
        with self.assertRaisesRegex(ValueError, "declared total params"):
            ladder.validate()

    def test_v42_preset_apis_are_removed(self):
        self.assertFalse(hasattr(V42Ladder, "production_ladder"))
        self.assertFalse(hasattr(rbuild, "StagePLedger"))
        self.assertFalse(hasattr(rbuild, "estimate_stage_p_cost"))
        self.assertFalse(hasattr(rbuild, "default_v42_data_sources"))

    def test_release_gate_reuses_v41_preservation(self):
        output = torch.tensor([[1.0, 2.0]])
        ladder = tiny_ladder()
        gate = v42_release_gate(output, output, 1.0, 1.0,
                                ladder=ladder, distributed=custom_distributed())
        self.assertTrue(gate["passed"])
        self.assertFalse(v42_release_gate(output, output, 1.0, 0.9,
                                          ladder=ladder)["passed"])
        self.assertFalse(v42_release_gate(output, output + 1,
                                          ladder=ladder)["passed"])


class V42DataTests(unittest.TestCase):
    def test_data_mix_is_explicitly_caller_defined(self):
        sources = [
            V42DataSource("mine", "my-org/my-data", weight=3.0),
            V42DataSource("other", "my-org/other-data", weight=0.5),
        ]
        stream = V42StreamingData(sources, loaders={
            "mine": lambda: iter([{"text": "a"}]),
            "other": lambda: iter([{"text": "b"}]),
        })
        self.assertEqual([source.dataset for source in stream.sources],
                         ["my-org/my-data", "my-org/other-data"])
        self.assertEqual(sorted(stream.rows()), ["a", "b"])

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
        batch = next(iter(stream.batches(tiny_config())))
        self.assertEqual(batch[0].shape, (1, 16))
        self.assertEqual(batch[1].shape, (1, 16))

    def test_production_data_requires_license_and_filter(self):
        with self.assertRaises(ValueError):
            V42StreamingData([V42DataSource("code", "caller/data")], production=True)


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

    def test_production_final_stage_requires_sharding_and_release_evidence(self):
        ladder = tiny_ladder(production=True)
        runner = V42ScratchTrainer(ladder, device="cpu", log_fn=None)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "sharding wrap_fn"):
                runner.run_stage(1, batches(), root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
