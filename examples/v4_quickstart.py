"""Minimal R-Build v4 training + exact-AR generation check.

Random weights demonstrate mechanics only. Train/distill the drafter before
using the acceptance or latency numbers for a release decision.
"""
import torch
from rbuild import RBuildConfig, RBuildModel, v4_release_gate

cfg = RBuildConfig()
cfg.model.vocab_size = 256
cfg.model.d_model = 64
cfg.model.max_seq_len = 64
cfg.model.n_heads = 4
cfg.model.n_kv_heads = 2
cfg.memory.key_dim = 32
cfg.memory.value_dim = 64
cfg.critic.max_loops = 2
cfg.draft_verify.d_model = 32
cfg.draft_verify.ffn_dim = 64
cfg.draft_verify.n_heads = 4

model = RBuildModel(cfg)
x = torch.randint(0, 256, (2, 32))
y = torch.randint(0, 256, (2, 32))
loss = model(x, targets=y)[1]
loss.backward()
print(f"joint main + draft loss: {float(loss.detach()):.4f}")

model.eval()
prompt = torch.randint(0, 256, (1, 8))
torch.manual_seed(42)
ar = model.generate(prompt, max_new_tokens=16, use_v4=False, top_p=1.0)
torch.manual_seed(42)
v4 = model.generate(prompt, max_new_tokens=16, use_v4=True, top_p=1.0)
print("AR == v4:", torch.equal(ar, v4))
print("decoder stats:", model.last_v4_stats)

# Replace these toy values with real frozen-set acceptance and canary scores.
report = v4_release_gate(
    acceptance_rate=model.last_v4_stats["acceptance_rate"],
    ar_output=ar,
    v4_output=v4,
    canary_before=0.80,
    canary_after=0.80,
)
print("release report:", report)
if report["passed"]:
    model.freeze_v4(report)
else:
    print("Not released: distill more and rerun the frozen-set gates.")
