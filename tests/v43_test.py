"""R-Build V4.3 efficiency + correctness tests (standard-library unittest)."""
import unittest

import torch
import torch.nn.functional as F

from rbuild import RBuildConfig, RBuildModel
from rbuild.layers.attention import MoDRouter
from rbuild.layers.moe import FineGrainedMoE
from rbuild.model import Block, _TokenSubsetBlock
from rbuild.train import Trainer


def tiny_config():
    cfg = RBuildConfig()
    cfg.model.max_seq_len = 64
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.critic.min_loops = 1
    cfg.critic.max_loops = 2
    cfg.noting.enabled = False
    cfg.actuation.enabled = False
    cfg.memory.enabled = True
    cfg.memory.key_dim = 32
    cfg.memory.value_dim = 64
    cfg.draft_verify.enabled = False
    cfg.outgen.enabled = False
    cfg.parallel.n_stages = 2
    cfg.parallel.n_branches = 2
    cfg.parallel.n_experts = 4
    cfg.cache_loop.n_loops = 2
    cfg.validate()
    return cfg


class MoDContextTests(unittest.TestCase):
    def test_routed_subset_is_one_causal_sequence(self):
        """capacity=1 routes every token, so MoD must equal the full block."""
        torch.manual_seed(0)
        cfg = tiny_config()
        blk = Block(cfg, ffn_kind="dense").eval()
        router = MoDRouter(cfg.model.d_model, capacity=1.0).eval()
        x = torch.randn(2, 10, cfg.model.d_model)
        with torch.no_grad():
            got = router(x, _TokenSubsetBlock(blk))
            full = blk(x)[0]
            gate = torch.sigmoid(router.score(x))
            want = x + gate * (full - x)
        self.assertTrue(torch.allclose(got, want, atol=1e-5),
                        "MoD-routed tokens must attend to each other")

    def test_routed_tokens_see_earlier_context(self):
        torch.manual_seed(1)
        cfg = tiny_config()
        blk = Block(cfg, ffn_kind="dense").eval()
        router = MoDRouter(cfg.model.d_model, capacity=1.0).eval()
        x = torch.randn(1, 8, cfg.model.d_model)
        x2 = x.clone()
        x2[0, 0] += 3.0                                  # perturb only token 0
        with torch.no_grad():
            a = router(x, _TokenSubsetBlock(blk))
            b = router(x2, _TokenSubsetBlock(blk))
        self.assertGreater((a[0, 5] - b[0, 5]).abs().max().item(), 1e-4)


class GeometryTests(unittest.TestCase):
    def test_unshared_loop_line_matches_counter(self):
        cfg = tiny_config()
        cfg.cache_loop.share_loop_weights = False
        cfg.cache_loop.n_layers = 3
        cfg.cache_loop.n_loops = 2
        model = RBuildModel(cfg)
        self.assertEqual(len(model.cache_loop.loop_line), 2)
        self.assertTrue(all(len(line) == 3 for line in model.cache_loop.loop_line))
        actual = sum(p.numel() for p in model.parameters())
        self.assertEqual(actual, cfg.count_parameters()["total_params"])


class MoEDispatchTests(unittest.TestCase):
    def test_grouped_dispatch_matches_reference(self):
        torch.manual_seed(2)
        moe = FineGrainedMoE(32, n_experts=6, top_k=2, expert_dim=16, n_shared=1)
        x = torch.randn(3, 7, 32, requires_grad=True)
        got = moe(x)
        # reference: v4.2 masked per-expert loop
        flat = x.reshape(-1, 32)
        probs = F.softmax(moe.router(flat).float(), dim=-1)
        top_p, top_i = probs.topk(2, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)
        ref = torch.zeros_like(flat)
        for e in range(6):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if tok.numel():
                ref.index_add_(0, tok, moe.experts[e](flat[tok]) * top_p[tok, slot].unsqueeze(-1))
        ref = ref + sum(s(flat) for s in moe.shared) / len(moe.shared)
        self.assertTrue(torch.allclose(got.reshape(-1, 32), ref, atol=1e-5))
        got.sum().backward()
        self.assertIsNotNone(x.grad)


class FusedCETests(unittest.TestCase):
    def test_fused_ce_matches_materialised_ce(self):
        torch.manual_seed(3)
        cfg = tiny_config()
        cfg.train.ce_chunk_tokens = 7                     # ragged chunks
        model = RBuildModel(cfg).train()
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        y[0, :3] = -100
        cfg.train.fused_lm_ce = True
        torch.manual_seed(9)
        logits, fused = model(x, targets=y)
        self.assertIsNone(logits)
        fused.backward()
        g_fused = model.embed.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        cfg.train.fused_lm_ce = False
        torch.manual_seed(9)
        logits, plain = model(x, targets=y)
        self.assertIsNotNone(logits)
        plain.backward()
        self.assertTrue(torch.allclose(fused, plain, atol=1e-5))
        self.assertTrue(torch.allclose(g_fused, model.embed.weight.grad, atol=1e-5))


class KVCacheTests(unittest.TestCase):
    def test_cached_hidden_matches_full_encode(self):
        torch.manual_seed(4)
        cfg = tiny_config()
        model = RBuildModel(cfg).eval()
        ids = torch.randint(0, 256, (1, 12))
        with torch.no_grad():
            full, _, _ = model._encode(ids, stable_inference=True)
            caches = {}
            first = model._encode_cached(ids[:, :8], caches, 0)
            steps = [first]
            for t in range(8, 12):
                steps.append(model._encode_cached(ids[:, t:t + 1], caches, t))
            inc = torch.cat(steps, dim=1)
        self.assertTrue(torch.allclose(full, inc, atol=1e-4))

    def test_cached_generation_matches_exact_generation(self):
        torch.manual_seed(5)
        cfg = tiny_config()
        model = RBuildModel(cfg).eval()
        prompt = torch.tensor([[5, 6, 7, 8]])
        torch.manual_seed(11)
        exact = model.generate(prompt, max_new_tokens=10, top_k=1, kv_cache=False)
        torch.manual_seed(11)
        cached = model.generate(prompt, max_new_tokens=10, top_k=1, kv_cache=True)
        self.assertTrue(torch.equal(exact, cached))
        self.assertGreater(model.last_kv_cache_bytes, 0)


class TrainerTests(unittest.TestCase):
    def test_fit_with_v43_stack(self):
        torch.manual_seed(6)
        cfg = tiny_config()
        cfg.train.grad_checkpoint = True
        cfg.train.grad_accum = 2
        cfg.train.batch_size = 2
        cfg.train.max_steps = 3
        cfg.train.peak_tflops_per_gpu = 1.0
        cfg.model.max_seq_len = 16
        model = RBuildModel(cfg)
        batches = [(torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16)))
                   for _ in range(4)]
        logs = []
        out = Trainer(model, cfg, device="cpu", log_fn=logs.append).fit(batches)
        self.assertEqual(len(out["loss"]), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(out["loss"]))))
        self.assertTrue(any("MFU" in line for line in logs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
