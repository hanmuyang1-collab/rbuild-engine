"""R-Build v4.2: interactive progressive training beside the V4.1 route.

V4.2 adds orchestration, validation, data admission, and distributed launch
contracts. It deliberately adds no modules or parameters to ``RBuildModel`` and
does not alter V4.1's exact base-model delegation.

There are no built-in V4.2 weight, ladder, hardware, or data-mix presets. Caller
scripts define every stage and runtime plan. A shape change between stages
requires an explicit growth function; weights are never silently reset.
"""

from __future__ import annotations

import json
import os
import queue
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from .config import RBuildConfig
from .data import _collate
from .model import RBuildModel
from .train import Trainer, _autocast_ctx, _unwrap
from .v41 import v41_release_gate


# --------------------------------------------------------------------------- #
# Caller-sized memory ledger
# --------------------------------------------------------------------------- #

@dataclass
class V42MemoryLedger:
    """Aggregate first-order training-residency ledger in decimal GB."""

    total_params: int
    gpu_count: int
    gpu_vram_gb: int
    fp32_master_bytes: int = 4
    adam_moment_bytes: int = 8
    bf16_weight_bytes: int = 2
    gradient_bytes: int = 2

    @property
    def bytes_per_param(self) -> int:
        return (self.fp32_master_bytes + self.adam_moment_bytes
                + self.bf16_weight_bytes + self.gradient_bytes)

    @property
    def aggregate_vram_gb(self) -> float:
        return self.gpu_count * self.gpu_vram_gb

    @property
    def training_resident_gb(self) -> float:
        return self.total_params * self.bytes_per_param / 1e9

    @property
    def remaining_activation_budget_gb(self) -> float:
        return self.aggregate_vram_gb - self.training_resident_gb

    @property
    def training_resident_per_gpu_gb(self) -> float:
        return self.training_resident_gb / self.gpu_count

    def validate(self) -> None:
        if self.total_params <= 0 or self.gpu_count <= 0 or self.gpu_vram_gb <= 0:
            raise ValueError("Memory-ledger sizes must be positive")
        if min(self.fp32_master_bytes, self.adam_moment_bytes,
               self.bf16_weight_bytes, self.gradient_bytes) < 0:
            raise ValueError("Residency byte counts cannot be negative")
        if self.remaining_activation_budget_gb < 0:
            raise ValueError("Training residency exceeds aggregate GPU memory")

    def to_dict(self) -> dict:
        self.validate()
        return {
            **asdict(self),
            "bytes_per_param": self.bytes_per_param,
            "aggregate_vram_gb": self.aggregate_vram_gb,
            "training_resident_gb": self.training_resident_gb,
            "training_resident_per_gpu_gb": self.training_resident_per_gpu_gb,
            "remaining_activation_budget_gb": self.remaining_activation_budget_gb,
            "remaining_budget_purpose": (
                "activations, KV, fast weights, draft head, and CUA replay; "
                "not additional weight residency"
            ),
        }


# --------------------------------------------------------------------------- #
# Caller-defined progressive ladder
# --------------------------------------------------------------------------- #

@dataclass
class V42Stage:
    """One scratch-ladder stage; ``config=None`` means still open."""

    name: str
    config: Optional[RBuildConfig] = None
    total_params: Optional[int] = None
    active_params: Optional[int] = None
    max_steps: Optional[int] = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("V4.2 stage names must be nonempty")
        if self.config is not None:
            self.config.validate()
            counts = self.config.count_parameters()
            if self.total_params is not None and self.total_params != counts["total_params"]:
                raise ValueError(f"Stage {self.name}: declared total params differ from its config")
        for value in (self.total_params, self.active_params, self.max_steps):
            if value is not None and value <= 0:
                raise ValueError(f"Stage {self.name}: sizes and max_steps must be positive")

    def resolved_counts(self) -> dict:
        if self.config is not None:
            return self.config.count_parameters()
        return {
            "total_params": self.total_params,
            "active_params_per_token": self.active_params,
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config": self.config.to_dict() if self.config is not None else None,
            "total_params": self.total_params,
            "active_params": self.active_params,
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_dict(cls, raw: Mapping) -> "V42Stage":
        config = raw.get("config")
        return cls(
            name=raw["name"],
            config=RBuildConfig.from_dict(config) if config is not None else None,
            total_params=raw.get("total_params"),
            active_params=raw.get("active_params"),
            max_steps=raw.get("max_steps"),
        )


@dataclass
class V42Ladder:
    """A caller-defined progressive ladder; no stage names or sizes are preset."""

    stages: List[V42Stage]
    production: bool = False

    def validate(self, allow_open: bool = True) -> None:
        if not self.stages:
            raise ValueError("V4.2 requires at least one caller-defined stage")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("V4.2 stage names must be unique")
        for stage in self.stages:
            stage.validate()
        if not allow_open and any(stage.config is None for stage in self.stages):
            open_names = [stage.name for stage in self.stages if stage.config is None]
            raise ValueError(f"Open V4.2 stages need explicit model configs: {open_names}")

    def index(self, name_or_index) -> int:
        if isinstance(name_or_index, int):
            if not 0 <= name_or_index < len(self.stages):
                raise IndexError(
                    f"V4.2 stage index must be in [0, {len(self.stages) - 1}]"
                )
            return name_or_index
        for index, stage in enumerate(self.stages):
            if stage.name == name_or_index:
                return index
        raise KeyError(f"Unknown V4.2 stage {name_or_index!r}")

    def save(self, path: str) -> None:
        self.validate()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"version": "4.2.0", "production": self.production,
                 "stages": [stage.to_dict() for stage in self.stages]},
                handle, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "V42Ladder":
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        ladder = cls([V42Stage.from_dict(item) for item in raw["stages"]],
                     production=bool(raw.get("production", False)))
        ladder.validate()
        return ladder

    def report(self) -> str:
        self.validate()
        lines = ["R-Build v4.2 caller-defined scratch ladder", "=" * 52]
        for index, stage in enumerate(self.stages):
            counts = stage.resolved_counts()
            total = counts.get("total_params")
            active = counts.get("active_params_per_token")
            total_text = "open" if total is None else f"{total / 1e9:.3f}B"
            active_text = "open" if active is None else f"{active / 1e9:.3f}B"
            lines.append(
                f"  {index + 1}. {stage.name}: total={total_text}, active={active_text}, "
                f"config={'ready' if stage.config is not None else 'open'}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Caller-defined distributed plan
# --------------------------------------------------------------------------- #

@dataclass
class V42DistributedConfig:
    """Distributed dimensions supplied by the caller, with no hardware preset."""

    nodes: int
    gpus_per_node: int
    gpu_vram_gb: int
    tensor_parallel: int
    n_experts: int
    expert_parallel: int
    expert_replicas: int
    expert_placements_per_node: int
    max_context: int
    flash_attention_3: bool = True
    full_activation_recompute: bool = True
    context_parallel: bool = True
    kv_tiers: Sequence[str] = ("gpu", "cpu", "nvme")
    fast_weight_tiers: Sequence[str] = ("gpu", "cpu", "nvme")
    interconnect: str = ""
    intranode_interconnect: str = ""

    @property
    def gpu_count(self) -> int:
        return self.nodes * self.gpus_per_node

    def validate(self) -> None:
        if min(self.nodes, self.gpus_per_node, self.gpu_vram_gb,
               self.tensor_parallel, self.n_experts, self.expert_parallel,
               self.expert_replicas, self.expert_placements_per_node,
               self.max_context) <= 0:
            raise ValueError("Distributed dimensions must be positive")
        if self.gpus_per_node % self.tensor_parallel:
            raise ValueError("TP groups must fit wholly inside each node")
        if self.expert_parallel != self.n_experts * self.expert_replicas:
            raise ValueError("EP must equal caller-defined experts times replicas")
        if self.nodes * self.expert_placements_per_node != self.expert_parallel:
            raise ValueError("Expert placements per node must cover EP exactly")
        if tuple(self.kv_tiers) != ("gpu", "cpu", "nvme"):
            raise ValueError("KV must use the GPU/CPU/NVMe hierarchy")
        if tuple(self.fast_weight_tiers) != ("gpu", "cpu", "nvme"):
            raise ValueError("Fast weights must use the GPU/CPU/NVMe hierarchy")
        if self.max_context >= 10_000_000 and not (
                self.flash_attention_3 and self.full_activation_recompute
                and self.context_parallel):
            raise ValueError(
                "10M context requires FA3, full activation recompute, and context sharding"
            )

    def expert_placements(self) -> Dict[int, List[dict]]:
        """Place caller-defined experts and replicas round-robin across nodes."""
        self.validate()
        placement = {node: [] for node in range(self.nodes)}
        for expert in range(self.n_experts):
            for replica_index in range(self.expert_replicas):
                node = (expert * self.expert_replicas + replica_index) % self.nodes
                placement[node].append({
                    "expert": expert,
                    "role": "primary" if replica_index == 0 else "replica",
                })
        return placement

    def memory_ledger(self, total_params: int) -> V42MemoryLedger:
        """Size memory from the caller's model, never from a built-in weight count."""
        self.validate()
        ledger = V42MemoryLedger(total_params, self.gpu_count, self.gpu_vram_gb)
        ledger.validate()
        return ledger

    def to_dict(self) -> dict:
        self.validate()
        raw = asdict(self)
        raw["kv_tiers"] = list(self.kv_tiers)
        raw["fast_weight_tiers"] = list(self.fast_weight_tiers)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping) -> "V42DistributedConfig":
        plan = cls(**dict(raw))
        plan.validate()
        return plan

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "V42DistributedConfig":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def nccl_environment(self) -> Dict[str, str]:
        """Portable defaults; fabric-specific tuning remains user-overridable."""
        return {
            "NCCL_IB_DISABLE": "0",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
        }

    def torchrun_command(
        self,
        script: str,
        node_rank: int,
        master_addr: str,
        script_args: Sequence[str] = (),
        master_port: int = 29500,
    ) -> List[str]:
        self.validate()
        if not 0 <= node_rank < self.nodes:
            raise ValueError(f"node_rank must be in [0, {self.nodes - 1}]")
        if not master_addr:
            raise ValueError("master_addr is required")
        return [
            "torchrun", f"--nnodes={self.nodes}", f"--node-rank={node_rank}",
            f"--nproc-per-node={self.gpus_per_node}",
            f"--master-addr={master_addr}", f"--master-port={master_port}",
            script, *list(script_args),
        ]

    def initialize_process_group(self, timeout_minutes: int = 60) -> int:
        """Initialize NCCL from torchrun environment and return the global rank."""
        self.validate()
        import torch.distributed as dist

        for key, value in self.nccl_environment().items():
            os.environ.setdefault(key, value)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != self.gpu_count:
            raise RuntimeError(
                f"Distributed plan expects WORLD_SIZE={self.gpu_count}, got {world_size}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The production distributed plan requires CUDA")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                timeout=__import__("datetime").timedelta(minutes=timeout_minutes),
            )
        return dist.get_rank()


@dataclass
class V42ThroughputProfile:
    """Measured throughput and price assumptions for one cost column."""

    gpu_price_per_hour: float
    achieved_tflops_per_gpu: float
    efficiency: float = 1.0
    flops_per_token: Optional[float] = None

    def validate(self) -> None:
        if min(self.gpu_price_per_hour, self.achieved_tflops_per_gpu,
               self.efficiency) <= 0:
            raise ValueError("Cost and throughput assumptions must be positive")
        if self.flops_per_token is not None and self.flops_per_token <= 0:
            raise ValueError("flops_per_token must be positive")


def estimate_v42_cost(
    total_tokens: int,
    total_params: int,
    active_params: int,
    naive: V42ThroughputProfile,
    optimized: V42ThroughputProfile,
    distributed: V42DistributedConfig,
) -> Dict[str, dict]:
    """Compute naive/optimized costs from caller-supplied sizes and hardware.

    The first-order defaults are dense 6*total_params and 6*active_params
    FLOPs/token. Either profile can override these with measured end-to-end
    FLOPs/token. No model size, GPU count, grant, or price is assumed.
    """
    if total_tokens <= 0 or total_params <= 0 or active_params <= 0:
        raise ValueError("Token and parameter counts must be positive")
    if active_params > total_params:
        raise ValueError("active_params cannot exceed total_params")
    distributed.validate()
    result = {}
    for name, profile, default_params in (
        ("naive", naive, total_params),
        ("optimized", optimized, active_params),
    ):
        profile.validate()
        flops_per_token = profile.flops_per_token or 6.0 * default_params
        denominator = (
            distributed.gpu_count * profile.achieved_tflops_per_gpu * 1e12
            * profile.efficiency * 3600.0
        )
        hours = total_tokens * flops_per_token / denominator
        result[name] = {
            "hours": hours,
            "cost_usd": hours * distributed.gpu_count * profile.gpu_price_per_hour,
            "flops_per_token": flops_per_token,
            "total_tokens": int(total_tokens),
            "gpu_count": distributed.gpu_count,
        }
    result["speedup_x"] = result["naive"]["hours"] / result["optimized"]["hours"]
    result["assumption"] = (
        "first-order compute only; use measured FLOPs/token for shared layers, "
        "activation recompute, communication, data stalls, and checkpointing"
    )
    return result


# --------------------------------------------------------------------------- #
# Streaming data admission and concurrent mixing
# --------------------------------------------------------------------------- #

@dataclass
class V42DataSource:
    name: str
    dataset: str
    weight: float = 1.0
    config: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    license: str = "review-required"
    license_approved: bool = False
    pii_policy: str = "admission-filter-required"
    admission_filter: Optional[Callable[[Mapping], bool]] = field(
        default=None, repr=False, compare=False
    )

    def validate(self, production: bool = False) -> None:
        if not self.name or not self.dataset or self.weight <= 0:
            raise ValueError("Data sources need a name, dataset, and positive weight")
        if not self.license or not self.pii_policy:
            raise ValueError(f"{self.name}: license and PII policy are manifest requirements")
        if production and not self.license_approved:
            raise ValueError(f"{self.name}: dataset license has not been approved")
        if production and self.admission_filter is None:
            raise ValueError(f"{self.name}: production ingestion requires an admission filter")


class _V42BatchIterable:
    def __init__(self, owner: "V42StreamingData", cfg: RBuildConfig,
                 batch: Optional[int], seq: Optional[int]):
        self.owner, self.cfg, self.batch, self.seq = owner, cfg, batch, seq

    def __iter__(self):
        batch = self.batch or self.cfg.train.batch_size
        seq = self.seq or self.cfg.model.max_seq_len
        samples, warned = [], {}
        for text in self.owner.rows():
            samples.append((text, None, None))
            if len(samples) == batch:
                yield _collate(self.cfg, samples, seq, warned)
                samples = []
        if samples:
            yield _collate(self.cfg, samples, seq, warned)


class V42StreamingData:
    """Concurrent streaming ingestion with admission before tokenization.

    ``loaders`` is a test/local escape hatch mapping source names to callables
    returning row iterables. Without it, Hugging Face ``load_dataset`` is always
    called with ``streaming=True``. Each source has a bounded queue, so a fast
    source cannot buffer an unbounded dataset in RAM.
    """

    def __init__(
        self,
        sources: Sequence[V42DataSource],
        seed: int = 42,
        queue_size: int = 32,
        loaders: Optional[Mapping[str, Callable[[], Iterable[Mapping]]]] = None,
        production: bool = False,
    ):
        if not sources:
            raise ValueError("V4.2 data mix cannot be empty")
        self.sources = list(sources)
        self.seed = seed
        self.queue_size = queue_size
        self.loaders = dict(loaders or {})
        self.production = production
        for source in self.sources:
            source.validate(production=production)

    def _source_rows(self, source: V42DataSource) -> Iterable[str]:
        if source.name in self.loaders:
            rows = self.loaders[source.name]()
        else:
            from datasets import load_dataset
            rows = load_dataset(
                source.dataset, name=source.config, split=source.split, streaming=True
            )
        for row in rows:
            # License/PII admission is part of ingestion, never a post-hoc pass.
            if source.admission_filter is not None and not source.admission_filter(row):
                continue
            text = row.get(source.text_column) or row.get("content")
            if text:
                yield str(text)

    def rows(self) -> Iterable[str]:
        rng = random.Random(self.seed)
        stop = threading.Event()
        queues = [queue.Queue(maxsize=self.queue_size) for _ in self.sources]

        def worker(index: int, source: V42DataSource) -> None:
            def put(item) -> bool:
                while not stop.is_set():
                    try:
                        queues[index].put(item, timeout=0.1)
                        return True
                    except queue.Full:
                        pass
                return False

            try:
                for text in self._source_rows(source):
                    if not put((True, text)):
                        return
                put((False, None))
            except BaseException as exc:
                put((False, exc))

        threads = [
            threading.Thread(target=worker, args=(index, source), daemon=True)
            for index, source in enumerate(self.sources)
        ]
        for thread in threads:
            thread.start()
        active = set(range(len(self.sources)))
        weights = [source.weight for source in self.sources]
        try:
            while active:
                index = rng.choices(range(len(self.sources)), weights=weights, k=1)[0]
                if index not in active:
                    continue
                ok, value = queues[index].get()
                if not ok:
                    active.remove(index)
                    if isinstance(value, BaseException):
                        raise value
                    continue
                yield value
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=1.0)

    def batches(self, cfg: RBuildConfig, batch: Optional[int] = None,
                seq: Optional[int] = None) -> Iterable:
        return _V42BatchIterable(self, cfg, batch, seq)


# --------------------------------------------------------------------------- #
# Resumable scratch training
# --------------------------------------------------------------------------- #

class V42StageTrainer(Trainer):
    """The existing native trainer with exact optimizer/scheduler-step resume."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0

    def fit(self, batches: Iterable, max_steps: Optional[int] = None,
            save_every: int = 0, save_dir: Optional[str] = None) -> dict:
        cfg = self.cfg
        max_steps = max_steps or cfg.train.max_steps
        self.model.train()
        accumulation, started, history = 0, time.time(), []
        initial_step = self.step
        data_iter = iter(batches)
        while self.step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(batches)
                batch = next(data_iter)
            if len(batch) == 4:
                x, y, images, out_targets = batch
                if images is not None:
                    images = images.to(self.device)
                out_targets = {key: value.to(self.device)
                               for key, value in out_targets.items()}
            elif len(batch) == 3:
                x, y, images = batch
                images = images.to(self.device)
                out_targets = None
            else:
                x, y = batch
                images, out_targets = None, None
            x, y = x.to(self.device), y.to(self.device)
            lr_factor = self.sched.set(self.step)
            with _autocast_ctx(cfg.train.precision, self.device_type):
                _, loss = self.model(x, targets=y, images=images,
                                     out_targets=out_targets)
                scaled_loss = loss / cfg.train.grad_accum
            scaled_loss.backward()
            accumulation += 1
            if accumulation % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                self.step += 1
                history.append(float(loss.detach()) * cfg.train.grad_accum)
                if self.step == 1 or self.step % 10 == 0:
                    tokens = (cfg.train.batch_size * cfg.train.grad_accum
                              * cfg.model.max_seq_len * (self.step - initial_step))
                    tokens_per_second = tokens / max(1e-9, time.time() - started)
                    self.log(
                        f"step {self.step}/{max_steps} loss {history[-1]:.4f} "
                        f"lr_x{lr_factor:.3f} ~{tokens_per_second:,.0f} tok/s"
                    )
                if cfg.train.self_train_every > 0 and self.step % cfg.train.self_train_every == 0:
                    self.self_train_step()
                if save_every > 0 and save_dir and self.step % save_every == 0:
                    self.save_checkpoint(save_dir)
        return {"loss": history, "steps": self.step}

    def save_checkpoint(self, path: str) -> None:
        super().save_checkpoint(path)
        torch.save(
            {"step": self.step,
             "optimizers": [optimizer.state_dict() for optimizer in self.optimizers]},
            os.path.join(path, "v42_trainer.pt"),
        )

    def load_training_state(self, path: str) -> None:
        state = torch.load(
            os.path.join(path, "v42_trainer.pt"), map_location="cpu", weights_only=False
        )
        if len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("Checkpoint optimizer count does not match this run")
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        self.step = int(state["step"])


class V42ScratchTrainer:
    """Build and advance a caller-defined ladder without implicit reinitialization.

    ``growth_fn(previous_model, current_model)`` is mandatory when adjacent
    stages have different state-dict geometry. It must perform the intended
    progressive expansion and return an optional metadata dictionary. With
    identical geometry, the default transfer is a strict full-state load.
    """

    def __init__(
        self,
        ladder: V42Ladder,
        device: Optional[str] = None,
        log_fn: Optional[Callable[[str], None]] = print,
        wrap_fn: Optional[Callable] = None,
        growth_fn: Optional[Callable[[RBuildModel, RBuildModel], Optional[dict]]] = None,
        release_evaluator: Optional[Callable] = None,
        base_seed: int = 4200,
    ):
        ladder.validate(allow_open=False)
        self.ladder = ladder
        self.device = device
        self.log = log_fn or (lambda *_: None)
        self.wrap_fn = wrap_fn
        self.growth_fn = growth_fn
        self.release_evaluator = release_evaluator
        self.base_seed = base_seed

    @staticmethod
    def _strict_transfer(previous: RBuildModel, current: RBuildModel) -> dict:
        old_state, new_state = previous.state_dict(), current.state_dict()
        if old_state.keys() != new_state.keys() or any(
                old_state[key].shape != new_state[key].shape for key in old_state):
            raise ValueError(
                "Adjacent V4.2 stages have different weight geometry. Supply an "
                "explicit growth_fn; refusing to discard learned weights."
            )
        current.load_state_dict(old_state, strict=True)
        return {"mode": "strict-identical-geometry", "tensors": len(old_state)}

    def run_stage(
        self,
        name_or_index,
        batches: Iterable,
        output_dir: str,
        resume_checkpoint: Optional[str] = None,
        previous_checkpoint: Optional[str] = None,
        save_every: int = 0,
        stop_step: Optional[int] = None,
    ) -> dict:
        index = self.ladder.index(name_or_index)
        stage = self.ladder.stages[index]
        assert stage.config is not None
        if self.ladder.production and index == len(self.ladder.stages) - 1:
            if self.wrap_fn is None:
                raise RuntimeError(
                    "The production final stage requires an explicit sharding wrap_fn"
                )
            if self.release_evaluator is None:
                raise RuntimeError(
                    "The production final stage requires a V4.1-compatible release evaluator"
                )

        stage_dir = os.path.join(output_dir, stage.name)
        if resume_checkpoint is not None:
            model = Trainer.load_checkpoint(resume_checkpoint, device=self.device)
            if model.cfg.to_dict() != stage.config.to_dict():
                raise ValueError("Resume checkpoint config differs from the ladder stage")
            transfer = {"mode": "exact-resume", "checkpoint": resume_checkpoint}
        else:
            torch.manual_seed(self.base_seed + index)
            model = RBuildModel(stage.config)
            if index > 0:
                previous_checkpoint = previous_checkpoint or os.path.join(
                    output_dir, self.ladder.stages[index - 1].name
                )
                previous = Trainer.load_checkpoint(previous_checkpoint, device="cpu")
                if self.growth_fn is None:
                    transfer = self._strict_transfer(previous, model)
                else:
                    transfer = self.growth_fn(previous, model) or {"mode": "custom-growth"}
                transfer["checkpoint"] = previous_checkpoint
            else:
                transfer = {"mode": "from-scratch"}

        trainer = V42StageTrainer(
            model, stage.config, device=self.device, log_fn=self.log,
            wrap_fn=self.wrap_fn,
        )
        if resume_checkpoint is not None:
            trainer.load_training_state(resume_checkpoint)
        planned_steps = stage.max_steps or stage.config.train.max_steps
        max_steps = stop_step or planned_steps
        if max_steps > planned_steps:
            raise ValueError("stop_step cannot exceed the stage's planned WSD horizon")
        result = trainer.fit(
            batches, max_steps=max_steps, save_every=save_every, save_dir=stage_dir
        )
        trainer.save_checkpoint(stage_dir)

        release = {"status": "not_run"}
        if self.release_evaluator is not None:
            release = self.release_evaluator(
                model=_unwrap(trainer.model), stage=stage, stage_index=index,
                checkpoint=stage_dir,
            )
            if not release.get("passed", False):
                raise RuntimeError(f"V4.2 release gate rejected Stage {stage.name}: {release}")
        metadata = {
            "version": "4.2.0",
            "route": "from-scratch",
            "stage": stage.name,
            "stage_index": index,
            "steps": trainer.step,
            "weight_transfer": transfer,
            "release_gate": release,
            "counts": stage.config.count_parameters(),
        }
        with open(os.path.join(stage_dir, "v42_stage.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return {"checkpoint": stage_dir, "training": result, "metadata": metadata}

    def run(self, batches_by_stage: Mapping[str, Iterable], output_dir: str,
            start_stage=0, resume_checkpoint: Optional[str] = None) -> List[dict]:
        start = self.ladder.index(start_stage)
        results = []
        for index in range(start, len(self.ladder.stages)):
            stage = self.ladder.stages[index]
            results.append(self.run_stage(
                index, batches_by_stage[stage.name], output_dir,
                resume_checkpoint=resume_checkpoint if index == start else None,
            ))
        return results


def v42_release_gate(
    base_output,
    adapted_output,
    canary_before: float = 0.0,
    canary_after: float = 0.0,
    canary_tolerance: float = 0.0,
    ladder: Optional[V42Ladder] = None,
    distributed: Optional[V42DistributedConfig] = None,
) -> dict:
    """Apply V4.1's no-regression gate plus V4.2 plan invariants."""
    preservation = v41_release_gate(
        base_output, adapted_output, canary_before, canary_after, canary_tolerance
    )
    ladder_ok, distributed_ok = True, True
    errors = []
    if ladder is not None:
        try:
            ladder.validate()
        except (ValueError, AssertionError) as exc:
            ladder_ok = False
            errors.append(str(exc))
    if distributed is not None:
        try:
            distributed.validate()
        except (ValueError, AssertionError) as exc:
            distributed_ok = False
            errors.append(str(exc))
    return {
        **preservation,
        "passed": bool(preservation["passed"] and ladder_ok and distributed_ok),
        "ladder_ok": ladder_ok,
        "distributed_ok": distributed_ok,
        "errors": errors,
    }
