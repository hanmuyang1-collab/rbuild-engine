# R-Build v4.3 — GPU and memory efficiency

v4.3 keeps every v4.2 API and checkpoint format. It fixes correctness bugs that
were also wasting compute, removes host-device syncs and redundant memory, and
adds opt-in knobs for compile, 8-bit optimizer states and KV-cached decoding.

License: v4.3.0 and later are under the **Velandris License 1.0** (Han Muyang).
Versions up to and including 4.2.0 stay available under Apache-2.0.

## Measured results (A100-SXM4-40GB, bf16 autocast, torch 2.14)

Benchmark: `benchmarks/gpu_bench.py` — 252M-param R-Build (d_model 1024,
2-layer cache loop x3, 3 parallel stages x 2 branches, 16 experts top-2),
seq 2048, AdamW, 10 timed steps after warm-up. Same script, same GPU, v4 branch
vs v4.3.

| Setting | v4.2 tok/s | v4.3 tok/s | Speed-up | v4.2 peak | v4.3 peak | Memory saved |
|---|---|---|---|---|---|---|
| batch 4, vocab 32k | 20,490 | 37,495 | 1.83x | 13.19 GiB | 10.85 GiB | 18% |
| batch 8, vocab 32k | 23,537 | 45,878 | 1.95x | 22.95 GiB | 16.71 GiB | 27% |
| batch 8, grad checkpoint | 18,679 | 35,933 | 1.92x | 19.58 GiB | 8.79 GiB | 55% |
| batch 4, vocab 131k | 15,853 | 21,743 | 1.37x | 21.58 GiB | 16.91 GiB | 22% |

Single-sequence decode (512-token prompt): v4.2 12.9 tok/s, v4.3 exact path
21.8 tok/s, v4.3 KV cache 32.3 tok/s (2.5x).

These are small-model, single-GPU numbers. Multi-GPU (DDP `no_sync`, FSDP2
Muon) and B200 MFU still need to be measured in a real pilot run.

## Problems fixed

| # | Problem in v4.2 | Fix in v4.3 |
|---|---|---|
| 1 | **MoD blocks had no attention.** `MoDRouter` flattened the routed tokens into B*k sequences of length 1, so every routed token only attended to itself. | Routed tokens are sorted and run as one causal sequence per batch row at their true RoPE positions (`positions=`, `pos_limit=`). |
| 2 | **Unshared cache-loop line built the wrong number of layers** (`range(n_loops)` instead of `range(n_layers)`), disagreeing with the parameter counter. | Uses `n_layers`; a test checks it against `param_count()`. |
| 3 | **"Chunked" CE still built the full (B, T, V) logits** and synced to the host every chunk. | Fused chunked LM-head + CE (`train.fused_lm_ce`, default on): logits are computed per chunk and recomputed in backward, never materialised. Training forward returns `logits=None`. Automatically off when draft co-training needs logits. |
| 4 | **Generation re-encoded the whole prefix every token** (O(T²)). | `model.kv_cache_generation=True` or `generate(kv_cache=True)`: incremental decode with per-layer K/V caches. Default stays on the v4 exact path. |
| 5 | **MoE dispatch synced to the host once per expert.** | Sort-based grouped dispatch, one `bincount().tolist()` per layer; aux loss accumulated in fp32. |
| 6 | **GQA copied K/V with `repeat_interleave` every call.** | SDPA `enable_gqa=True` (fallback kept for old torch). |
| 7 | **RoPE tables rebuilt in every attention call** of every block and loop. | Cached per (head_dim, theta, device, dtype). |
| 8 | **RMSNorm was four kernels.** | `F.rms_norm` fused kernel for fp32 activations. |
| 9 | **Cache-loop activations were never checkpointed**, so loops multiplied activation memory. | `train.grad_checkpoint` now covers loop iterations. |
| 10 | **Muon ran Newton-Schulz in fp32 and broke on FSDP2 sharded params.** | bf16 Newton-Schulz on CUDA (`train.muon_bf16`); DTensor params are gathered with `full_tensor()` and re-sharded. |
| 11 | **AdamW was the unfused default.** | Fused AdamW on CUDA (foreach on CPU); optional bitsandbytes 8-bit states (`train.optimizer_8bit`). |
| 12 | **DDP all-reduced every micro-batch** during gradient accumulation. | `no_sync()` on non-final micro-batches. |
| 13 | **Loss `.item()` every step** stalled the GPU. | Losses stay on device and are read back only at log steps; H2D copies are `non_blocking`. |
| 14 | **Bad memory config crashed deep in the first forward.** | `validate()` checks `memory.value_dim == model.d_model`. |

## New knobs

```python
cfg.train.fused_lm_ce = True          # default; chunked LM-head + CE
cfg.train.compile = False             # torch.compile(dynamic=True) in Trainer
cfg.train.optimizer_8bit = False      # needs bitsandbytes
cfg.train.muon_bf16 = True            # CUDA only; CPU stays fp32
cfg.train.peak_tflops_per_gpu = None  # e.g. 989 (H100) / 2250 (B200) -> logs MFU
cfg.model.kv_cache_generation = False # opt-in KV-cached AR decode
```

TF32 matmuls are enabled automatically on CUDA. The trainer logs MFU (from
`active_params_per_token`) and peak GiB when `peak_tflops_per_gpu` is set.

## Caveats

- **MoD semantics changed.** Fix 1 makes MoD blocks real attention blocks. v4.2
  checkpoints load, but models trained with MoD on will behave differently and
  should be fine-tuned briefly. Losses in the benchmark are not comparable.
- **KV-cache decode is float-tolerant, not byte-exact**, versus the re-encode
  path. The exact path remains the default and the v4 draft-verify contract is
  unchanged.
- `torch.compile` is off by default; enable it after checking your config
  compiles cleanly.
- Tests: `tests/v43_test.py` (8 tests) plus all earlier suites pass on CPU.
