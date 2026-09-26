# R-Build v2.0

**A fully user-modifiable, open-source LLM architecture + training-speed engine.**
Every knob is yours: programmatically, through an interactive widget panel in
Colab/Jupyter, or a console prompt anywhere else.

## Architecture (v2)

```
tokens ─► embed ─► CACHE-LOOP LINE ─► PARALLEL STAGE 1 ─► ... ─► PARALLEL STAGE N ─► head
                    │  single line of   │  n_branches parallel blocks per stage;
                    │  layers, looped   │  outputs bundled (gate / mean / concat)
                    │  n_loops times,   │  and pushed to the next parallel stage
                    │  pulling the      │
                    │  fast-weight      ▼
                    │  cache each loop  logits
                    ▼
            FastWeightMemory — delta-rule key→value matrix,
            gradient-free fact writes, saved with every checkpoint
```

- **Stage A — cache loop ("first set of layers in a single line, loops to pull cache")**:
  one line of blocks, applied `cache_loop.n_loops` times (shared weights by default),
  performing a delta-rule **read** against the fast-weight cache every
  `memory_read_every` loops and injecting it through a learned gate.
- **Stage B — parallel bundle stages ("multiple parallel layers ... bundle to push
  to the next set of parallel layers")**: each stage runs `n_branches` blocks in
  parallel over the same input and **bundles** them (`bundle_mode`: `gate`,
  `mean`, or `concat`), pushing the bundle onward. The final bundle produces logits.

Carried over from the R-Build optimized stack: **MoD routing** (top-p tokens per
block), **fine-grained MoE + dense-sized shared expert**, **Muon** optimizer,
**WSD** schedule, **chunked cross-entropy**, and the **fast-weight delta-rule
memory** (facts written without any training step).

## Install

```bash
pip install .                       # from this folder
pip install .[notebook]             # + ipywidgets for the interactive panel
```

## Use it — everything is modifiable

```python
from rbuild import RBuildConfig, RBuildModel, Trainer, preset

cfg = preset("s1")                  # counter-verified: 19.9B total / 3.3B active
cfg.parallel.n_branches = 8         # change anything
cfg.cache_loop.n_loops = 10
cfg.train.precision = "bf16"
print(cfg.report())                 # params + naive-vs-optimized cost, always both

model = RBuildModel(cfg)
assert sum(p.numel() for p in model.parameters()) == cfg.count_parameters()["total_params"]
```

### Interactive panel (Colab / Jupyter)

```python
from rbuild import interactive
ui = interactive.launch()           # widgets for every value, live report
# ... tweak, click "Build model" ...
model = ui.model
```

Outside notebooks the same call falls back to console prompts.

### Interactive generation + memory

```python
interactive.chat(model, encode, decode)
# /remember <text>   gradient-free fact write into the fast-weight cache
# /temp /topp /topk /maxn          live sampling control
# /forget            reset the cache      /config   live report
```

### Train

```python
trainer = Trainer(model, cfg)       # Muon + WSD + chunked CE + bf16/fp8 flag
trainer.fit(batches)                # batches yield (input_ids, targets)
trainer.save_checkpoint("ckpt/")    # weights + memory matrix + config, together
model2 = Trainer.load_checkpoint("ckpt/")
```

## Stage ladder presets (continued-training path)

| preset | total | active/token | shape |
|---|---|---|---|
| s1 | 19.9B | 3.3B | d=2048, 7 stages × 6 branches, 128 experts |
| s2 | 90.6B | 8.9B | d=2560, 6 × 9, 160 experts |
| s3 | 117.9B | 18.8B | d=4096 — wider per token, not deeper |
| s4 | 219.2B | 39.6B | d=5120, 12 × 8, 96 experts |
| s5 | 414.9B | 61.6B | d=6144, 9 × 11, 32 wide experts |

All presets are counter-verified against the v2 architecture; treat them as
starting points and retune in the panel.

Apache-2.0. Part of the GeoThinkAI R-Build project.
