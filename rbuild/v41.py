"""R-Build v4.1: non-destructive base-model conversion and training.

Any PyTorch/Hugging Face causal-LM architecture can become an R-Build system
without replacing its native layers. The base forward/generation path remains
byte-for-byte authoritative; R-Build critics and fast-weight memory are additive
capabilities. Continued pretraining and supervised fine-tuning reuse Muon, WSD,
gradient accumulation, mixed precision, and checkpoint persistence.

No base checkpoint is hardcoded. Every adapted model also exposes the universal
R-Build CUA action codec, action head, training loss, and guarded runtime.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Iterable, Mapping, Optional

import torch
import torch.nn as nn

from .actuation import ActionCodec, ActuationHead
from .config import TrainConfig
from .critics import CriticPanel
from .cua import CUAApplicationAdapter, ComputerUseAgent, freeze_for_cua as freeze_cua_parameters
from .memory import FastWeightMemory
from .optim import WSDScheduler, build_optimizer
from .train import _autocast_ctx, _unwrap
from .vision import EncoderlessScreenshotProjector


@dataclass
class V41Config:
    """Additive R-Build capabilities plus the original training speed stack."""

    hidden_size: Optional[int] = None
    memory_enabled: bool = True
    critic_enabled: bool = True
    n_critics: int = 4
    y_critics: int = 2
    critic_capacity_mult: float = 2.0
    critic_hidden: Optional[int] = None
    critic_threshold: float = 0.6
    memory_write_lr: float = 1.0
    memory_decay: float = 0.999
    memory_max_facts: int = 65536
    cua_vocab_size: Optional[int] = None
    cua_screen_grid: int = 64
    cua_scroll_steps: int = 8
    cua_screenshot_enabled: bool = False
    cua_screenshot_size: int = 64
    cua_screenshot_patch: int = 16
    cua_screenshot_channels: int = 3
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        if self.hidden_size is not None and self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.n_critics <= 0 or not 0 <= self.y_critics <= self.n_critics:
            raise ValueError("require n_critics > 0 and 0 <= y_critics <= n_critics")
        if self.critic_capacity_mult <= 0:
            raise ValueError("critic_capacity_mult must be positive")
        if self.cua_vocab_size is not None and self.cua_vocab_size <= 0:
            raise ValueError("cua_vocab_size must be positive")
        if self.cua_screen_grid <= 0 or self.cua_scroll_steps <= 0:
            raise ValueError("CUA screen_grid and scroll_steps must be positive")
        if (
            self.cua_screenshot_size <= 0
            or self.cua_screenshot_patch <= 0
            or self.cua_screenshot_channels <= 0
            or self.cua_screenshot_size % self.cua_screenshot_patch
        ):
            raise ValueError("CUA screenshot dimensions and patch size are invalid")
        if self.train.grad_accum <= 0 or self.train.max_steps <= 0:
            raise ValueError("grad_accum and max_steps must be positive")

    def resolved_hidden_size(self, base_model: nn.Module) -> int:
        if self.hidden_size is not None:
            return int(self.hidden_size)
        config = getattr(base_model, "config", None)
        for name in ("hidden_size", "d_model", "n_embd", "dim"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise ValueError(
            "Cannot infer the base hidden size; set V41Config.hidden_size explicitly"
        )

    def resolved_vocab_size(self, base_model: nn.Module) -> int:
        if self.cua_vocab_size is not None:
            return int(self.cua_vocab_size)
        config = getattr(base_model, "config", None)
        value = getattr(config, "vocab_size", None)
        if value is not None:
            return int(value)
        output_head = base_model.get_output_embeddings() if hasattr(
            base_model, "get_output_embeddings"
        ) else None
        if output_head is not None and hasattr(output_head, "out_features"):
            return int(output_head.out_features)
        for name in ("head", "lm_head", "embed_tokens", "embed"):
            module = getattr(base_model, name, None)
            if hasattr(module, "out_features"):
                return int(module.out_features)
            if hasattr(module, "num_embeddings"):
                return int(module.num_embeddings)
        raise ValueError(
            "Cannot infer the base vocabulary size; set V41Config.cua_vocab_size explicitly"
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "V41Config":
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        train = TrainConfig(**raw.pop("train", {}))
        names = {item.name for item in fields(cls)}
        return cls(train=train, **{key: value for key, value in raw.items() if key in names})


class RBuildAdaptedModel(nn.Module):
    """Wrap any base architecture without modifying its native computation.

    ``forward`` and ``generate`` are exact delegations. Critics and fast-weight
    memory never change base outputs unless a caller explicitly consumes their
    outputs, so disabling or wrapping capabilities cannot degrade the base model.
    """

    CONFIG_NAME = "rbuild_v41_config.json"
    ADAPTER_NAME = "rbuild_v41_adapter.pt"

    def __init__(self, base_model: nn.Module, config: Optional[V41Config] = None):
        super().__init__()
        self.config = config or V41Config()
        self.config.validate()
        self.base_model = base_model
        hidden = self.config.resolved_hidden_size(base_model)
        self.hidden_size = hidden
        self.memory = (
            FastWeightMemory(
                hidden,
                hidden,
                write_lr=self.config.memory_write_lr,
                decay=self.config.memory_decay,
                max_facts=self.config.memory_max_facts,
            )
            if self.config.memory_enabled
            else None
        )
        critic_hidden = self.config.critic_hidden or max(
            hidden, int(hidden * self.config.critic_capacity_mult)
        )
        self.critics = (
            CriticPanel(
                hidden,
                critic_hidden,
                self.config.n_critics,
                threshold=self.config.critic_threshold,
            )
            if self.config.critic_enabled
            else None
        )
        self.action_codec = ActionCodec(
            self.config.resolved_vocab_size(base_model),
            self.config.cua_screen_grid,
            self.config.cua_scroll_steps,
        )
        self.action_head = ActuationHead(hidden, self.action_codec.n_actions)
        self.cua = ComputerUseAgent(self, self.action_codec, self.action_head)
        self.screenshot_projector = (
            EncoderlessScreenshotProjector(
                hidden,
                self.config.cua_screenshot_size,
                self.config.cua_screenshot_patch,
                self.config.cua_screenshot_channels,
            )
            if self.config.cua_screenshot_enabled
            else None
        )
        self.cua_adapters = nn.ModuleDict()
        self._active_cua_adapter: Optional[str] = None

    @classmethod
    def from_pretrained(
        cls,
        base_model_id: str,
        config: Optional[V41Config] = None,
        auto_model_cls=None,
        **from_pretrained_kwargs,
    ) -> "RBuildAdaptedModel":
        """Convert a selected pretrained checkpoint; the base is caller-modifiable."""
        if auto_model_cls is None:
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise ImportError("Install rbuild-engine[train] for HF conversion") from exc
            auto_model_cls = AutoModelForCausalLM
        base = auto_model_cls.from_pretrained(base_model_id, **from_pretrained_kwargs)
        return cls(base, config)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def freeze_for_cua(self) -> None:
        """Train only the existing CUA head and opt-in CUA adapters/projector."""
        freeze_cua_parameters(self)

    def add_cua_adapter(self, name: str, rank: int = 8, alpha: float = 16.0) -> None:
        """Attach a zero-initialized application adapter to the shared CUA path."""
        if not name or name in self.cua_adapters:
            raise ValueError("CUA adapter names must be nonempty and unique")
        self.cua_adapters[name] = CUAApplicationAdapter(self.hidden_size, rank, alpha)

    def activate_cua_adapter(self, name: Optional[str] = None) -> None:
        if name is not None and name not in self.cua_adapters:
            raise KeyError(f"Unknown CUA application adapter: {name}")
        self._active_cua_adapter = name

    def transform_cua_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._active_cua_adapter is None:
            return hidden_states
        return self.cua_adapters[self._active_cua_adapter](hidden_states)

    def screenshot_hidden(self, screenshots: torch.Tensor) -> torch.Tensor:
        if self.screenshot_projector is None:
            raise RuntimeError("Enable V41Config.cua_screenshot_enabled for screenshot CUA")
        return self.screenshot_projector(screenshots)

    def freeze_base(self, frozen: bool = True) -> None:
        """Choose full base training or adapter-only fine-tuning at runtime."""
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(not frozen)

    @torch.no_grad()
    def remember_hidden(
        self,
        hidden_states: torch.Tensor,
        values: Optional[torch.Tensor] = None,
        require_critic: bool = True,
    ) -> int:
        """Critic-gate and delta-rule-write hidden facts; return write count."""
        if self.memory is None:
            raise RuntimeError("fast-weight memory is disabled")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states has the wrong final dimension")
        values = hidden_states if values is None else values
        flat_keys = hidden_states.detach().reshape(-1, self.hidden_size)
        flat_values = values.detach().reshape(-1, self.hidden_size)
        if require_critic and self.critics is not None:
            _, votes = self.critics(hidden_states.detach())
            approved = (votes >= self.config.y_critics).reshape(-1)
            flat_keys = flat_keys[approved]
            flat_values = flat_values[approved]
        if flat_keys.shape[0] == 0:
            return 0
        self.memory.write(flat_keys, flat_values)
        return int(flat_keys.shape[0])

    def recall_hidden(self, keys: torch.Tensor) -> torch.Tensor:
        if self.memory is None:
            raise RuntimeError("fast-weight memory is disabled")
        return self.memory.read(keys)

    def adapter_state_dict(self) -> dict:
        return {
            "memory": self.memory.state_dict() if self.memory is not None else None,
            "critics": self.critics.state_dict() if self.critics is not None else None,
            "action_head": self.action_head.state_dict(),
            "screenshot_projector": (
                self.screenshot_projector.state_dict()
                if self.screenshot_projector is not None else None
            ),
            "cua_adapters": self.cua_adapters.state_dict(),
            "cua_adapter_specs": {
                name: {"rank": adapter.rank, "alpha": adapter.alpha}
                for name, adapter in self.cua_adapters.items()
            },
            "active_cua_adapter": self._active_cua_adapter,
        }

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        base_path = os.path.join(path, "base")
        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(base_path)
            base_format = "pretrained"
        else:
            os.makedirs(base_path, exist_ok=True)
            torch.save(self.base_model.state_dict(), os.path.join(base_path, "model.pt"))
            base_format = "state_dict"
        torch.save(self.adapter_state_dict(), os.path.join(path, self.ADAPTER_NAME))
        self.config.save(os.path.join(path, self.CONFIG_NAME))
        with open(os.path.join(path, "rbuild_v41_meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"base_format": base_format}, handle, indent=2)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        base_model: Optional[nn.Module] = None,
        device: str = "cpu",
        **from_pretrained_kwargs,
    ) -> "RBuildAdaptedModel":
        config = V41Config.load(os.path.join(path, cls.CONFIG_NAME))
        base_path = os.path.join(path, "base")
        if base_model is None:
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise ImportError("Pass base_model= or install rbuild-engine[train]") from exc
            base_model = AutoModelForCausalLM.from_pretrained(base_path, **from_pretrained_kwargs)
        else:
            state_path = os.path.join(base_path, "model.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=device, weights_only=False)
                base_model.load_state_dict(state)
        model = cls(base_model, config)
        adapter = torch.load(
            os.path.join(path, cls.ADAPTER_NAME), map_location=device, weights_only=False
        )
        if model.memory is not None and adapter["memory"] is not None:
            model.memory.load_state_dict(adapter["memory"])
        if model.critics is not None and adapter["critics"] is not None:
            model.critics.load_state_dict(adapter["critics"])
        if adapter.get("action_head") is not None:
            model.action_head.load_state_dict(adapter["action_head"])
        if model.screenshot_projector is not None and adapter.get("screenshot_projector") is not None:
            model.screenshot_projector.load_state_dict(adapter["screenshot_projector"])
        for name, spec in adapter.get("cua_adapter_specs", {}).items():
            model.add_cua_adapter(name, spec["rank"], spec["alpha"])
        if adapter.get("cua_adapters"):
            model.cua_adapters.load_state_dict(adapter["cua_adapters"])
        model.activate_cua_adapter(adapter.get("active_cua_adapter"))
        return model.to(device)


def _move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_batch(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_batch(item, device) for item in value)
    if isinstance(value, list):
        return [_move_batch(item, device) for item in value]
    return value


def _output_loss(output) -> torch.Tensor:
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, Mapping):
        loss = output.get("loss")
    if loss is None and isinstance(output, (tuple, list)) and output:
        loss = output[0]
    if not isinstance(loss, torch.Tensor):
        raise ValueError("The base model must return a loss when labels are supplied")
    return loss


class RBuildBaseTrainer:
    """Continued-pretraining/fine-tuning trainer for any adapted base model."""

    def __init__(
        self,
        model: nn.Module,
        config: V41Config,
        device: Optional[str] = None,
        log_fn=print,
        wrap_fn=None,
    ):
        config.validate()
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if wrap_fn is not None:
            self.model = wrap_fn(self.model)
        self.log = log_fn or (lambda *_: None)
        self.optimizers = build_optimizer(self.model, config)
        base_lrs = [[group["lr"] for group in opt.param_groups] for opt in self.optimizers]
        self.scheduler = WSDScheduler(
            self.optimizers,
            base_lrs,
            config.train.warmup_steps,
            config.train.max_steps,
            config.train.cooldown_frac,
        )
        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        self.step = 0

    def fit(
        self,
        batches: Iterable,
        max_steps: Optional[int] = None,
        save_every: int = 0,
        save_dir: Optional[str] = None,
    ) -> dict:
        """Train on HF-style mappings or ``(input_ids, labels)`` tuples."""
        max_steps = max_steps or self.config.train.max_steps
        self.model.train()
        history, accumulation, started = [], 0, time.time()
        data_iter = iter(batches)
        while self.step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(batches)
                batch = next(data_iter)
            batch = _move_batch(batch, self.device)
            if isinstance(batch, Mapping):
                inputs = dict(batch)
            else:
                input_ids, labels = batch[:2]
                inputs = {"input_ids": input_ids, "labels": labels}
            lr_factor = self.scheduler.set(self.step)
            with _autocast_ctx(self.config.train.precision, self.device_type):
                loss = _output_loss(self.model(**inputs))
                scaled_loss = loss / self.config.train.grad_accum
            scaled_loss.backward()
            accumulation += 1
            if accumulation % self.config.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip
                )
                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                self.step += 1
                history.append(float(loss.detach()))
                if self.step == 1 or self.step % 10 == 0:
                    tokens = inputs["input_ids"].numel()
                    self.log(
                        f"step {self.step}/{max_steps}  loss {history[-1]:.4f}  "
                        f"lr_x{lr_factor:.3f}  batch_tokens {tokens}  "
                        f"elapsed {time.time() - started:.1f}s"
                    )
                if save_every > 0 and save_dir and self.step % save_every == 0:
                    self.save_checkpoint(save_dir)
        return {"loss": history, "steps": self.step}

    def fit_cua(
        self,
        batches: Iterable,
        max_steps: Optional[int] = None,
        save_every: int = 0,
        save_dir: Optional[str] = None,
    ) -> dict:
        """Train the shared action head on verified hidden/action replay batches.

        Call ``RBuildAdaptedModel.freeze_for_cua()`` before constructing this
        trainer for the low-VRAM frozen-base path.
        """
        raw = _unwrap(self.model)
        if not hasattr(raw, "cua"):
            raise ValueError("fit_cua requires a model exposing model.cua")
        max_steps = max_steps or self.config.train.max_steps
        self.model.train()
        history, accumulation, started = [], 0, time.time()
        data_iter = iter(batches)
        while self.step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(batches)
                batch = next(data_iter)
            inputs = dict(_move_batch(batch, self.device))
            lr_factor = self.scheduler.set(self.step)
            with _autocast_ctx(self.config.train.precision, self.device_type):
                loss = raw.cua.action_loss(
                    inputs["hidden_states"], inputs["action_token_ids"]
                )
                scaled_loss = loss / self.config.train.grad_accum
            scaled_loss.backward()
            accumulation += 1
            if accumulation % self.config.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip
                )
                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                self.step += 1
                history.append(float(loss.detach()))
                if self.step == 1 or self.step % 10 == 0:
                    supervised = int((inputs["action_token_ids"] != -100).sum())
                    self.log(
                        f"cua step {self.step}/{max_steps}  loss {history[-1]:.4f}  "
                        f"lr_x{lr_factor:.3f}  actions {supervised}  "
                        f"elapsed {time.time() - started:.1f}s"
                    )
                if save_every > 0 and save_dir and self.step % save_every == 0:
                    self.save_checkpoint(save_dir)
        return {"loss": history, "steps": self.step}

    def save_checkpoint(self, path: str) -> None:
        model = _unwrap(self.model)
        if isinstance(model, RBuildAdaptedModel):
            model.save_checkpoint(path)
        else:
            os.makedirs(path, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(os.path.join(path, "base"))
            else:
                os.makedirs(os.path.join(path, "base"), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(path, "base", "model.pt"))
            self.config.save(os.path.join(path, RBuildAdaptedModel.CONFIG_NAME))
        torch.save(
            {
                "step": self.step,
                "optimizers": [opt.state_dict() for opt in self.optimizers],
            },
            os.path.join(path, "trainer.pt"),
        )
        self.log(f"V4.1 checkpoint saved -> {path}")

    def load_training_state(self, path: str) -> None:
        state = torch.load(os.path.join(path, "trainer.pt"), map_location="cpu", weights_only=False)
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        self.step = int(state["step"])


def continued_pretraining_batches(
    tokenizer,
    texts: Iterable[str],
    seq_len: int = 4096,
    batch_size: int = 1,
    add_eos: bool = True,
):
    """Pack raw documents into ordinary causal-LM CPT batches indefinitely."""
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    while True:
        buffer, pending = [], []
        produced = False
        for text in texts:
            if not text:
                continue
            buffer.extend(tokenizer.encode(text, add_special_tokens=False))
            if add_eos and getattr(tokenizer, "eos_token_id", None) is not None:
                buffer.append(tokenizer.eos_token_id)
            while len(buffer) >= seq_len:
                ids = torch.tensor(buffer[:seq_len], dtype=torch.long)
                buffer = buffer[seq_len:]
                pending.append(ids)
                if len(pending) == batch_size:
                    stacked = torch.stack(pending)
                    yield {
                        "input_ids": stacked,
                        "labels": stacked.clone(),
                        "attention_mask": torch.ones_like(stacked),
                    }
                    pending = []
                    produced = True
        if pending:
            stacked = torch.stack(pending)
            yield {
                "input_ids": stacked,
                "labels": stacked.clone(),
                "attention_mask": torch.ones_like(stacked),
            }
            produced = True
        if not produced:
            return


def supervised_fine_tuning_batches(
    tokenizer,
    examples: Iterable,
    seq_len: int = 4096,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int = 0,
):
    """Yield SFT batches with prompt labels masked to -100."""
    if seq_len <= 1 or batch_size <= 0:
        raise ValueError("seq_len must exceed 1 and batch_size must be positive")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    rng = random.Random(seed)
    while True:
        produced = False
        if shuffle and isinstance(examples, (list, tuple)):
            order = list(range(len(examples)))
            rng.shuffle(order)
            source = (examples[index] for index in order)
        else:
            source = iter(examples)
        pending = []
        for example in source:
            produced = True
            prompt, response = _prompt_response(example)
            prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            response_ids = list(tokenizer.encode(response, add_special_tokens=False))
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None:
                response_ids.append(eos_id)
            if len(response_ids) >= seq_len:
                response_ids = response_ids[:seq_len]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[: seq_len - len(response_ids)]
            ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids.copy()
            padding = seq_len - len(ids)
            pending.append(
                {
                    "input_ids": torch.tensor(ids + [pad_id] * padding, dtype=torch.long),
                    "labels": torch.tensor(labels + [-100] * padding, dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [1] * len(ids) + [0] * padding, dtype=torch.long
                    ),
                }
            )
            if len(pending) == batch_size:
                yield {key: torch.stack([row[key] for row in pending]) for key in pending[0]}
                pending = []
        if pending:
            yield {key: torch.stack([row[key] for row in pending]) for key in pending[0]}
        if not produced:
            # Empty or exhausted non-reiterable sources must not spin forever.
            return


def _prompt_response(example):
    if isinstance(example, Mapping):
        return example["prompt"], example["response"]
    return example


def _outputs_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_outputs_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_outputs_equal(a, b) for a, b in zip(left, right))
    if hasattr(left, "logits") and hasattr(right, "logits"):
        return _outputs_equal(left.logits, right.logits)
    return left == right


def v41_release_gate(
    base_output,
    adapted_output,
    canary_before: float = 0.0,
    canary_after: float = 0.0,
    canary_tolerance: float = 0.0,
) -> dict:
    """Require exact disabled-path output and non-regressing capability canaries."""
    exact = _outputs_equal(base_output, adapted_output)
    canary_ok = float(canary_after) >= float(canary_before) - float(canary_tolerance)
    return {"passed": bool(exact and canary_ok), "exact_base_output": exact, "canary_ok": canary_ok}
