"""
R-Build trainer: Muon + WSD + precision stack + chunked CE, with the
fast-weight memory riding along inside every checkpoint.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable, Optional

import torch

from .config import RBuildConfig
from .model import RBuildModel
from .optim import Muon, WSDScheduler, build_optimizer


def _autocast_ctx(precision: str, device_type: str):
    if precision == "fp32" or device_type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False) if device_type == "cpu" \
            else torch.autocast(device_type=device_type, enabled=False)
    if precision == "fp8":
        # FP8 compute requires transformer-engine on Hopper+; on other
        # hardware R-Build keeps the *contract* (fp8 flag) and runs bf16,
        # so configs stay portable across Colab / Vast.ai / home GPUs.
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


class Trainer:
    def __init__(self, model: RBuildModel, cfg: RBuildConfig,
                 device: Optional[str] = None, log_fn: Optional[Callable[[str], None]] = print):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.log = log_fn or (lambda *_: None)
        self.optimizers = build_optimizer(self.model, cfg)
        base_lrs = [[g["lr"] for g in opt.param_groups] for opt in self.optimizers]
        self.sched = WSDScheduler(self.optimizers, base_lrs,
                                  cfg.train.warmup_steps, cfg.train.max_steps,
                                  cfg.train.cooldown_frac)
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

    # ------------------------------------------------------------------ #
    def fit(self, batches: Iterable, max_steps: Optional[int] = None) -> dict:
        """
        batches: iterable yielding (input_ids, targets) LongTensors of shape
        (B, T). Re-loops the iterable if it is shorter than max_steps.
        """
        cfg = self.cfg
        max_steps = max_steps or cfg.train.max_steps
        self.model.train()
        step, accum, t0, history = 0, 0, time.time(), []
        data_iter = iter(batches)
        while step < max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(batches)
                x, y = next(data_iter)
            x, y = x.to(self.device), y.to(self.device)

            lr_f = self.sched.set(step)
            with _autocast_ctx(cfg.train.precision, self.device_type):
                _, loss = self.model(x, targets=y)
                loss = loss / cfg.train.grad_accum
            loss.backward()
            accum += 1

            if accum % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                for opt in self.optimizers:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                history.append(float(loss.detach()) * cfg.train.grad_accum)
                if step % 10 == 0 or step == 1:
                    tok_s = (cfg.train.batch_size * cfg.train.grad_accum
                             * cfg.model.max_seq_len * step) / max(1e-9, time.time() - t0)
                    self.log(f"step {step}/{max_steps}  loss {history[-1]:.4f}  "
                             f"lr_x{lr_f:.3f}  ~{tok_s:,.0f} tok/s")
        return {"loss": history}

    # ------------------------------------------------------------------ #
    # checkpoints: model weights + fast-weight memory + config, together
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.cfg.save(os.path.join(path, "rbuild_config.json"))
        meta = {"memory_writes": int(self.model.memory.n_writes) if self.model.memory else 0}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.log(f"checkpoint saved -> {path} (memory matrix included)")

    @staticmethod
    def load_checkpoint(path: str, device: Optional[str] = None) -> "RBuildModel":
        cfg = RBuildConfig.load(os.path.join(path, "rbuild_config.json"))
        model = RBuildModel(cfg)
        state = torch.load(os.path.join(path, "model.pt"),
                           map_location=device or "cpu", weights_only=False)
        model.load_state_dict(state)
        return model
