"""
R-Build trainer: Muon + WSD + precision stack + chunked CE, with the
fast-weight memory riding along inside every checkpoint.

v3 adds `self_train_step()`: the consolidation half of non-separate
self-training. Notes that critics verified during normal running wait in a
low-RAM CPU buffer; this step replays them and aligns the model's native
fact-writing projections with what the critics approved — the slow weights
absorb what the fast weights already learned.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F

from .config import RBuildConfig
from .model import RBuildModel
from .optim import Muon, WSDScheduler, build_optimizer

from contextlib import contextmanager


@contextmanager
def full_state_dict_ctx(model):
    """Rank-0 full state dict for FSDP-wrapped models; no-op otherwise."""
    try:
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                            StateDictType, FullStateDictConfig)
        if isinstance(model, FSDP):
            with FSDP.state_dict_type(
                    model, StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                yield
            return
    except ImportError:
        pass
    yield


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


def _unwrap(model):
    raw = getattr(model, "module", model)
    return getattr(raw, "_fsdp_wrapped_module", raw)


class Trainer:
    def __init__(self, model: RBuildModel, cfg: RBuildConfig,
                 device: Optional[str] = None, log_fn: Optional[Callable[[str], None]] = print,
                 wrap_fn: Optional[Callable] = None):
        """
        wrap_fn: optional model wrapper applied after .to(device) and before
        the optimizers are built — e.g. FSDP(..., use_orig_params=True) for
        multi-GPU 20B+ runs. Optimizers then see original parameter shapes,
        so the Muon/AdamW split still works.
        """
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if wrap_fn is not None:
            self.model = wrap_fn(self.model)
        self.log = log_fn or (lambda *_: None)
        self.optimizers = build_optimizer(self.model, cfg)
        base_lrs = [[g["lr"] for g in opt.param_groups] for opt in self.optimizers]
        self.sched = WSDScheduler(self.optimizers, base_lrs,
                                  cfg.train.warmup_steps, cfg.train.max_steps,
                                  cfg.train.cooldown_frac)
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"

    # ------------------------------------------------------------------ #
    def fit(self, batches: Iterable, max_steps: Optional[int] = None,
            save_every: int = 0, save_dir: Optional[str] = None) -> dict:
        """
        batches: iterable yielding (input_ids, targets) LongTensors of shape
        (B, T). Re-loops the iterable if it is shorter than max_steps.
        save_every > 0 writes a checkpoint to save_dir every N steps.
        If train.self_train_every > 0, verified self-observed notes are
        consolidated into the slow weights every N steps.
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
                    halt = getattr(_unwrap(self.model).cache_loop, "last_halting", None)
                    extra = f"  loops {halt['mean_loops']:.1f}" if halt else ""
                    self.log(f"step {step}/{max_steps}  loss {history[-1]:.4f}  "
                             f"lr_x{lr_f:.3f}  ~{tok_s:,.0f} tok/s{extra}")
                if cfg.train.self_train_every > 0 and step % cfg.train.self_train_every == 0:
                    st_loss = self.self_train_step()
                    if st_loss is not None:
                        self.log(f"  self-train consolidation loss {st_loss:.4f}")
                if save_every > 0 and save_dir and step % save_every == 0:
                    self.save_checkpoint(save_dir)
        return {"loss": history}

    # ------------------------------------------------------------------ #
    # v3: consolidate critic-verified notes into the slow weights
    # ------------------------------------------------------------------ #
    def self_train_step(self) -> Optional[float]:
        """
        One consolidation step over the verified-note buffer: replay the
        hidden states the notes were taken from and train the model's
        native fact projections (the same path `remember()` uses) to
        reproduce the critic-verified (key, value) pairs. Uses
        train.self_train_lr, independent of the WSD schedule.
        Returns the loss, or None if the buffer is empty.
        """
        model = _unwrap(self.model)
        learner = getattr(model, "self_learner", None)
        if learner is None:
            return None
        batch = learner.buffer.sample(self.cfg.noting.self_train_batch)
        if batch is None:
            return None
        h, k, v, s = (t.to(self.device) for t in batch)

        was_training = model.training
        model.train()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)
        k_hat = model.fact_key_proj(h)
        v_hat = model.fact_value_proj(h)
        w = (s / s.sum().clamp_min(1e-9)).unsqueeze(-1)     # score-weighted
        loss = ((k_hat - k).pow(2) * w).sum() + ((v_hat - v).pow(2) * w).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
        # consolidation uses its own lr, then the schedule's lrs are restored
        saved = [[g["lr"] for g in opt.param_groups] for opt in self.optimizers]
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["lr"] = self.cfg.train.self_train_lr
        for opt in self.optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for opt, lrs in zip(self.optimizers, saved):
            for g, lr in zip(opt.param_groups, lrs):
                g["lr"] = lr
        if not was_training:
            model.eval()
        return float(loss.detach())

    # ------------------------------------------------------------------ #
    # checkpoints: model weights + fast-weight memory + config, together
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with full_state_dict_ctx(self.model):
            state = self.model.state_dict()
        # strip DDP "module." prefix so checkpoints load into a bare model
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        if state:  # rank0_only under FSDP gives {} on other ranks
            torch.save(state, os.path.join(path, "model.pt"))
        self.cfg.save(os.path.join(path, "rbuild_config.json"))
        inner = _unwrap(self.model)
        mem = getattr(inner, "memory", None)
        meta = {"memory_writes": int(mem.n_writes) if mem is not None else 0}
        learner = getattr(inner, "self_learner", None)
        if learner is not None:
            meta["self_learning"] = learner.stats()
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
