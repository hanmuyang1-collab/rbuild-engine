"""
R-Build v3 quickstart — every v3 feature in one tiny run.

    python examples/v3_quickstart.py

Covers: counter verification, critic-gated ACT halting, thinking modes
(built-in + user-created), critic-verified non-separate self-training,
encoderless VL + VaWU, and native actuation.
"""

import torch

from rbuild import RBuildConfig, RBuildModel, Trainer

torch.manual_seed(0)

# ---------------------------------------------------------------- config
cfg = RBuildConfig()                       # tiny default; scale with preset("s1")
cfg.model.max_seq_len = 128
cfg.critic.max_loops = 4
cfg.critic.n_critics = 4                   # X critics per panel
cfg.critic.y_critics = 2                   # halt when Y are satisfied
cfg.actuation.enabled = True               # native click/scroll/type tokens
cfg.actuation.screen_grid = 8              # coarse grid for the demo
cfg.vision.enabled = True                  # v3 encoderless VL
cfg.vision.mode = "encoderless"            # no vision encoder at all
cfg.vision.image_token_id = cfg.model.vocab_size - 1
cfg.vision.image_size = 32
cfg.vision.patch_size = 8
cfg.vision.vawu = True                     # whole-video summary tokens
cfg.train.max_steps = 20

print(cfg.report())

# ---------------------------------------------------------------- build
model = RBuildModel(cfg)
actual = sum(p.numel() for p in model.parameters())
counted = cfg.count_parameters()["total_params"]
assert actual == counted, f"counter mismatch: {actual} vs {counted}"
print(f"\n[counter verified] built {actual:,} params == counted {counted:,}")

# ---------------------------------------------------------------- thinking modes
model.thinking_mode.deep()                 # built-in
model.thinking_mode.create("exam", max_loops=3, y_critics=3, temperature=0.2)
mode = model.thinking_mode.exam()          # user-created, now native
print(f"[thinking] exam active: {mode}")

# ---------------------------------------------------------------- train step
B, T = 2, 32
x = torch.randint(0, cfg.model.vocab_size, (B, T))
logits, loss = model(x, targets=x)
print(f"[forward] logits {tuple(logits.shape)}  loss {loss.item():.4f}  "
      f"(vocab extended to {logits.shape[-1]} for action tokens)")
print(f"[halting] {model.cache_loop.last_halting}")

trainer = Trainer(model, cfg)
loss.backward()                            # grads exist; a real step:
trainer.self_train_step()                  # (buffer empty -> None, safe)

# ---------------------------------------------------------------- generate + self-learning
model.eval()
out = model.generate(x[:1], max_new_tokens=4)
print(f"[generate] {tuple(out.shape)}")
print(f"[self-learn] {model.self_learn_stats()}")

# ---------------------------------------------------------------- native actuation
codec = model.action_codec
demo = [codec.encode_click(0.25, 0.75), codec.encode_scroll(-2),
        codec.wait_id, codec.type_begin_id, 42, codec.type_end_id]
print(f"[actuation] {codec.decode_actions(demo)}")

# ---------------------------------------------------------------- encoderless VL + VaWU
frames = 3
tpi = (cfg.vision.image_size // cfg.vision.patch_size) ** 2
K = cfg.vision.vawu_tokens + frames * tpi  # VaWU tokens are prepended
ids = torch.full((1, K + 4), 5)
ids[0, 2:2 + K] = cfg.vision.image_token_id
imgs = torch.randn(1, frames, 3, cfg.vision.image_size, cfg.vision.image_size)
logits, _ = model(ids, images=imgs)
print(f"[vision] encoderless+VaWU forward ok: logits {tuple(logits.shape)}")

print("\nv3 quickstart complete.")
