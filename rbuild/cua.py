"""Universal computer-using-agent (CUA) contract for every R-Build model.

The same native ActionCodec/ActuationHead action space is used by the text
model, R-OutGen, and adapted pretrained models. Runtime verification, bounded
replay, policies, and auditing add no model parameters. Real OS effects always
remain behind an explicit caller-provided driver.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actuation import Action, ActionCodec, ActuationHead


class ConfirmationRequired(RuntimeError):
    """Raised before any side effect when irreversible work is unconfirmed."""


class PolicyViolation(RuntimeError):
    """Raised before execution when a CUA runtime policy rejects a batch."""


class PostconditionFailed(RuntimeError):
    """Raised when a driver reports that an expected postcondition is false."""


@dataclass
class CUADecision:
    """Native action-token ids, decoded actions, logits, and safety metadata."""

    token_ids: torch.Tensor
    actions: List[Action]
    logits: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    abstained: Optional[torch.Tensor] = None
    needs_confirmation: Optional[torch.Tensor] = None
    verification: Optional[Dict[str, Any]] = None


class CUAPolicy:
    """Application-specific, parameter-free action policy.

    Validation is whole-batch and runs before the first side effect. Rate-limit
    state advances only after successful execution.
    """

    def __init__(
        self,
        *,
        allowed_kinds: Optional[Sequence[str]] = None,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
        max_abs_scroll: Optional[int] = None,
        max_text_length: Optional[int] = None,
        blocked_text: Sequence[str] = (),
        confirmation_kinds: Sequence[str] = (),
        min_interval_seconds: float = 0.0,
    ):
        self.allowed_kinds = frozenset(allowed_kinds or ())
        self.x_bounds = tuple(map(float, x_bounds))
        self.y_bounds = tuple(map(float, y_bounds))
        self.max_abs_scroll = max_abs_scroll
        self.max_text_length = max_text_length
        self.blocked_text = tuple(text.casefold() for text in blocked_text)
        self.confirmation_kinds = frozenset(confirmation_kinds)
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_execution: Optional[float] = None
        if self.x_bounds[0] > self.x_bounds[1] or self.y_bounds[0] > self.y_bounds[1]:
            raise ValueError("CUA policy coordinate bounds are reversed")
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")

    def requires_confirmation(self, actions: Sequence[Action]) -> bool:
        return any(action.kind in self.confirmation_kinds for action in actions)

    def validate(self, actions: Sequence[Action], now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else float(now)
        if (
            self._last_execution is not None
            and now - self._last_execution < self.min_interval_seconds
        ):
            raise PolicyViolation("CUA action rate limit has not elapsed")
        for action in actions:
            if self.allowed_kinds and action.kind not in self.allowed_kinds:
                raise PolicyViolation(f"CUA action kind is not allowed: {action.kind}")
            if action.kind == "click":
                if action.x is None or action.y is None:
                    raise PolicyViolation("Click actions require coordinates")
                if not self.x_bounds[0] <= action.x <= self.x_bounds[1]:
                    raise PolicyViolation("Click x coordinate is outside the application policy")
                if not self.y_bounds[0] <= action.y <= self.y_bounds[1]:
                    raise PolicyViolation("Click y coordinate is outside the application policy")
            if (
                action.kind == "scroll"
                and self.max_abs_scroll is not None
                and abs(action.dy or 0) > self.max_abs_scroll
            ):
                raise PolicyViolation("Scroll magnitude exceeds the application policy")
            if action.kind == "type":
                text = action.text or ""
                if self.max_text_length is not None and len(text) > self.max_text_length:
                    raise PolicyViolation("Typed text exceeds the application length limit")
                folded = text.casefold()
                if any(blocked in folded for blocked in self.blocked_text):
                    raise PolicyViolation("Typed text matches an application deny rule")

    def mark_executed(self, now: Optional[float] = None) -> None:
        self._last_execution = time.monotonic() if now is None else float(now)


class PolicyDriver:
    """Driver wrapper carrying a policy that ComputerUseAgent enforces."""

    def __init__(self, driver, policy: CUAPolicy):
        self.driver = driver
        self.policy = policy

    def moveTo(self, x, y):
        return self.driver.moveTo(x, y)

    def click(self):
        return self.driver.click()

    def scroll(self, dy):
        return self.driver.scroll(dy)

    def write(self, text):
        return self.driver.write(text)

    def verify(self, expectation):
        return self.driver.verify(expectation)


class ShadowDriver:
    """A recording fake driver: no wrapped real driver method is ever called."""

    def __init__(self, real_driver=None):
        self.real_driver = real_driver
        self.events: List[tuple] = []
        self.postconditions: Dict[Any, bool] = {}

    def moveTo(self, x, y):
        self.events.append(("move", x, y))

    def click(self):
        self.events.append(("click",))

    def scroll(self, dy):
        self.events.append(("scroll", dy))

    def write(self, text):
        self.events.append(("type", text))

    def verify(self, expectation):
        if callable(expectation):
            return bool(expectation(self))
        return bool(self.postconditions.get(expectation, False))


class ScreenChangeDetector:
    """Downsampled screen-diff gate with minimum and maximum poll intervals."""

    def __init__(
        self,
        threshold: float = 0.01,
        min_interval: float = 0.0,
        max_interval: float = 5.0,
        signature_size: int = 16,
    ):
        if threshold < 0 or min_interval < 0 or max_interval <= 0 or signature_size <= 0:
            raise ValueError("Invalid screen-change detector configuration")
        self.threshold = float(threshold)
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.signature_size = int(signature_size)
        self.previous: Optional[torch.Tensor] = None
        self.last_observation: Optional[float] = None
        self.last_change: Optional[float] = None

    def _signature(self, screen) -> torch.Tensor:
        if isinstance(screen, (bytes, bytearray, memoryview)):
            digest = hashlib.blake2b(bytes(screen), digest_size=32).digest()
            return torch.tensor(list(digest), dtype=torch.float32) / 255.0
        value = torch.as_tensor(screen).detach().to(device="cpu", dtype=torch.float32)
        if value.ndim == 5:  # use the newest frame of a video observation
            value = value[:, -1]
        if value.ndim == 2:
            value = value[None, None]
        elif value.ndim == 3:
            value = value[None]
        if value.ndim != 4:
            raise ValueError("Screens must have shape (H,W), (C,H,W), (B,C,H,W), or video")
        value = value.mean(dim=1, keepdim=True)
        value = F.adaptive_avg_pool2d(value, (self.signature_size, self.signature_size))
        return value.flatten()

    def should_observe(self, screen, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        signature = self._signature(screen)
        if self.previous is None or self.last_observation is None:
            self.previous = signature
            self.last_observation = now
            self.last_change = None
            return True
        if signature.shape != self.previous.shape:
            change = 1.0
        else:
            change = float((signature - self.previous).abs().mean())
        self.last_change = change
        elapsed = now - self.last_observation
        observe = elapsed >= self.max_interval or (
            elapsed >= self.min_interval and change >= self.threshold
        )
        if observe:
            self.previous = signature
            self.last_observation = now
        return observe


class CUAReplayLedger:
    """Bounded in-memory audit trail with optional rotating JSONL persistence."""

    def __init__(
        self,
        max_records: int = 1024,
        path: Optional[str] = None,
        max_file_bytes: int = 8 * 1024 * 1024,
    ):
        if max_records <= 0 or max_file_bytes <= 0:
            raise ValueError("Replay limits must be positive")
        self.records = deque(maxlen=max_records)
        self.path = path
        self.max_file_bytes = int(max_file_bytes)

    @staticmethod
    def _actions(record: Mapping[str, Any]) -> List[Action]:
        names = {"kind", "x", "y", "dy", "text", "irreversible"}
        return [Action(**{key: value for key, value in item.items() if key in names})
                for item in record["actions"]]

    def append(
        self,
        actions: Sequence[Action],
        *,
        confirmed: bool,
        irreversible: bool,
        driver: str,
        status: str,
        postcondition=None,
        metadata: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> dict:
        record = {
            "timestamp": time.time(),
            "actions": [asdict(action) for action in actions],
            "confirmed": bool(confirmed),
            "irreversible": bool(irreversible),
            "driver": driver,
            "status": status,
            "postcondition": postcondition if isinstance(postcondition, (str, int, float, bool)) else None,
            "metadata": dict(metadata or {}),
            "error": error,
        }
        self.records.append(record)
        if self.path:
            self._append_jsonl(record)
        return record

    def _append_jsonl(self, record: Mapping[str, Any]) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.path) and os.path.getsize(self.path) >= self.max_file_bytes:
            os.replace(self.path, self.path + ".1")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def load_jsonl(self, path: Optional[str] = None) -> List[dict]:
        source = path or self.path
        if not source or not os.path.exists(source):
            return []
        with open(source, encoding="utf-8") as handle:
            loaded = [json.loads(line) for line in handle if line.strip()]
        self.records.extend(loaded)
        return loaded

    def replay(
        self,
        agent: "ComputerUseAgent",
        index: int,
        driver,
        *,
        confirmed: bool = False,
    ) -> List[Action]:
        """Replay recorded actions exactly, without sampling new model outputs."""
        record = self.records[index]
        actions = self._actions(record)
        irreversible = bool(record["irreversible"]) or any(a.irreversible for a in actions)
        return agent.execute(
            actions,
            driver,
            confirmed=confirmed,
            irreversible=irreversible,
            postcondition=record.get("postcondition"),
            metadata={"replay_of": index},
        )


@dataclass
class CUATrajectory:
    """A supervised hidden-state/action-token trajectory."""

    hidden_states: torch.Tensor
    action_token_ids: torch.Tensor
    verified: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.hidden_states.ndim == 2:
            self.hidden_states = self.hidden_states.unsqueeze(0)
        if self.action_token_ids.ndim == 1:
            self.action_token_ids = self.action_token_ids.unsqueeze(0)
        if self.hidden_states.ndim != 3 or self.action_token_ids.ndim != 2:
            raise ValueError("Trajectory tensors must be (B,T,D) and (B,T)")
        if self.hidden_states.shape[:2] != self.action_token_ids.shape:
            raise ValueError("Trajectory hidden states and labels have different shapes")
        if self.hidden_states.shape[0] != 1:
            raise ValueError("Store one trajectory per CUATrajectory; batch in the replay buffer")


class CUATrajectoryBuffer:
    """Verified-only, bounded CPU-fp16 trajectory replay."""

    def __init__(self, max_trajectories: int = 1024, max_bytes: int = 2 * 1024**3):
        if max_trajectories <= 0 or max_bytes <= 0:
            raise ValueError("Replay-buffer limits must be positive")
        self.max_trajectories = int(max_trajectories)
        self.max_bytes = int(max_bytes)
        self.items = deque()
        self.stored_bytes = 0

    def __len__(self):
        return len(self.items)

    def add(self, trajectory: CUATrajectory) -> None:
        if not trajectory.verified:
            raise ValueError("Unverified CUA trajectories cannot enter training replay")
        hidden = trajectory.hidden_states.detach().to(device="cpu", dtype=torch.float16)
        labels = trajectory.action_token_ids.detach().to(device="cpu", dtype=torch.long)
        item = CUATrajectory(hidden, labels, True, dict(trajectory.metadata))
        size = hidden.numel() * hidden.element_size() + labels.numel() * labels.element_size()
        while self.items and (
            len(self.items) >= self.max_trajectories or self.stored_bytes + size > self.max_bytes
        ):
            removed = self.items.popleft()
            self.stored_bytes -= (
                removed.hidden_states.numel() * removed.hidden_states.element_size()
                + removed.action_token_ids.numel() * removed.action_token_ids.element_size()
            )
        if size > self.max_bytes:
            raise ValueError("One CUA trajectory exceeds the replay byte budget")
        self.items.append(item)
        self.stored_bytes += size

    def batches(self, batch_size: int = 1, shuffle: bool = True, seed: int = 0):
        """Yield padded replay batches; padding labels are -100 and ignored."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        order = list(range(len(self.items)))
        if shuffle:
            random.Random(seed).shuffle(order)
        for start in range(0, len(order), batch_size):
            selected = [self.items[index] for index in order[start:start + batch_size]]
            width = max(item.hidden_states.shape[1] for item in selected)
            dim = selected[0].hidden_states.shape[-1]
            hidden = torch.zeros(len(selected), width, dim, dtype=torch.float32)
            labels = torch.full((len(selected), width), -100, dtype=torch.long)
            for row, item in enumerate(selected):
                length = item.hidden_states.shape[1]
                hidden[row, :length] = item.hidden_states[0].float()
                labels[row, :length] = item.action_token_ids[0]
            yield {"hidden_states": hidden, "action_token_ids": labels}


class CUAApplicationAdapter(nn.Module):
    """Zero-initialized low-rank residual adapter over existing CUA hidden states."""

    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or alpha <= 0:
            raise ValueError("CUA adapter dimensions and alpha must be positive")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.down = nn.Linear(hidden_size, self.rank, bias=False)
        self.up = nn.Linear(self.rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(self.down(hidden_states)) * (self.alpha / self.rank)


RECOVERY_EVENTS = (
    "unexpected_dialog",
    "scrolled_target",
    "loading_delay",
    "stale_screenshot",
    "failed_click",
)


def build_recovery_curriculum(
    codec: ActionCodec,
    trajectories: Iterable[CUATrajectory],
    *,
    events: Sequence[str] = RECOVERY_EVENTS,
    seed: int = 0,
) -> List[CUATrajectory]:
    """Derive verified interruption/recovery variants without new model weights."""
    unknown = set(events) - set(RECOVERY_EVENTS)
    if unknown:
        raise ValueError(f"Unknown recovery events: {sorted(unknown)}")
    rng = random.Random(seed)
    output: List[CUATrajectory] = []
    for source in trajectories:
        if not source.verified:
            continue
        hidden = source.hidden_states
        labels = source.action_token_ids
        length = hidden.shape[1]
        for event in events:
            index = rng.randrange(length)
            if event == "scrolled_target":
                flat = labels[0].tolist()
                click_positions = [
                    i for i, token in enumerate(flat)
                    if codec.click_base <= token < codec.scroll_base
                ]
                if not click_positions:
                    continue
                index = click_positions[rng.randrange(len(click_positions))]
                action = codec.decode_actions([flat[index]])[0]
                shift = 1.0 / codec.screen_grid
                y = action.y + shift if action.y + shift < 1.0 else action.y - shift
                changed = labels.clone()
                changed[0, index] = codec.encode_click(action.x, y)
                variant_hidden, variant_labels = hidden.clone(), changed
            else:
                # A repeated observation paired with wait teaches bounded recovery,
                # after which the original verified trajectory resumes.
                recovery_hidden = hidden[:, index:index + 1].clone()
                recovery_label = torch.full(
                    (1, 1), codec.wait_id, dtype=torch.long, device=labels.device
                )
                variant_hidden = torch.cat(
                    [hidden[:, :index], recovery_hidden, hidden[:, index:]], dim=1
                )
                variant_labels = torch.cat(
                    [labels[:, :index], recovery_label, labels[:, index:]], dim=1
                )
            output.append(CUATrajectory(
                variant_hidden,
                variant_labels,
                verified=True,
                metadata={**source.metadata, "recovery_event": event, "recovery_index": index},
            ))
    return output


class ComputerUseAgent:
    """Shared observe/propose/decode/train/verify/execute interface."""

    def __init__(self, model, codec: ActionCodec, action_head: ActuationHead):
        if codec is None or action_head is None:
            raise ValueError("Every R-Build model must provide a CUA codec and action head")
        self.model = model
        self.codec = codec
        self.action_head = action_head
        self.restored_from_legacy = False
        self.ledger = CUAReplayLedger()

    def _prepare_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        transform = getattr(self.model, "transform_cua_hidden", None)
        return transform(hidden_states) if transform is not None else hidden_states

    def decode(self, token_ids) -> List[Action]:
        """Decode reserved action tokens; ordinary text tokens are ignored."""
        return self.codec.decode_actions(token_ids)

    @torch.no_grad()
    def propose(
        self,
        hidden_states: torch.Tensor,
        position: int = -1,
        *,
        max_entropy: Optional[float] = None,
        min_confidence: Optional[float] = None,
        on_uncertain: str = "wait",
    ) -> CUADecision:
        """Propose actions with optional entropy/confidence abstention.

        ``on_uncertain='wait'`` emits the existing wait action. ``'confirm'``
        retains the proposal but requires confirmation before ``act()`` executes.
        These thresholds are independent of thinking effort and safety policy.
        """
        if on_uncertain not in {"wait", "confirm"}:
            raise ValueError("on_uncertain must be 'wait' or 'confirm'")
        if hidden_states.ndim not in (2, 3):
            raise ValueError("hidden_states must have shape (B, D) or (B, T, D)")
        if hidden_states.shape[-1] != self.action_head.proj.in_features:
            raise ValueError("hidden_states has the wrong final dimension")
        hidden_states = self._prepare_hidden(hidden_states)
        logits = self.action_head(hidden_states)
        selected = logits[:, position] if logits.ndim == 3 else logits
        probabilities = F.softmax(selected.float(), dim=-1)
        log_probabilities = F.log_softmax(selected.float(), dim=-1)
        confidence, offsets = probabilities.max(dim=-1)
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        uncertain = torch.zeros_like(confidence, dtype=torch.bool)
        if max_entropy is not None:
            uncertain |= entropy > float(max_entropy)
        if min_confidence is not None:
            uncertain |= confidence < float(min_confidence)
        abstained = uncertain & (on_uncertain == "wait")
        needs_confirmation = uncertain & (on_uncertain == "confirm")
        token_ids = offsets + self.codec.vocab_size
        if bool(abstained.any()):
            token_ids = torch.where(
                abstained, torch.full_like(token_ids, self.codec.wait_id), token_ids
            )
        actions = self.decode(token_ids.reshape(-1))
        return CUADecision(
            token_ids=token_ids,
            actions=actions,
            logits=selected,
            confidence=confidence.detach(),
            entropy=entropy.detach(),
            abstained=abstained,
            needs_confirmation=needs_confirmation,
        )

    def _critic_verdict(self, candidate_hidden: torch.Tensor):
        batch, candidates, dim = candidate_hidden.shape
        flat = candidate_hidden.reshape(batch * candidates, 1, dim)
        if hasattr(self.model, "critics_verify"):
            satisfaction, votes = self.model.critics_verify(flat)
        else:
            panel = getattr(self.model, "critics", None)
            if panel is None:
                raise RuntimeError("Critic-verified proposals require an existing critic panel")
            satisfaction, votes = panel(flat)
        return satisfaction.reshape(batch, candidates), votes.reshape(batch, candidates)

    @torch.no_grad()
    def propose_verified(
        self,
        hidden_states: torch.Tensor,
        *,
        candidates: int = 4,
        position: int = -1,
        min_votes: Optional[int] = None,
        min_satisfaction: float = 0.0,
        require_approval: bool = True,
        expected_postcondition=None,
    ) -> CUADecision:
        """Rank action candidates with existing critics and optional safe abstention."""
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        base = self.propose(hidden_states, position=position)
        prepared = self._prepare_hidden(hidden_states)
        selected_hidden = prepared[:, position] if prepared.ndim == 3 else prepared
        top_log_probability, top_offsets = F.log_softmax(
            base.logits.float(), dim=-1
        ).topk(min(candidates, self.codec.n_actions), dim=-1)
        candidate_hidden = (
            selected_hidden.unsqueeze(1) + self.action_head.proj.weight[top_offsets]
        )
        satisfaction, votes = self._critic_verdict(candidate_hidden)
        if min_votes is None:
            config = getattr(self.model, "config", None)
            native_cfg = getattr(self.model, "cfg", None)
            configured_votes = getattr(config, "y_critics", None)
            if configured_votes is None:
                configured_votes = getattr(getattr(native_cfg, "critic", None), "y_critics", 1)
            min_votes = int(configured_votes)
        approved = (votes >= int(min_votes)) & (satisfaction >= float(min_satisfaction))
        score = top_log_probability + satisfaction
        if require_approval:
            score = score.masked_fill(~approved, -float("inf"))
        selected_index = score.argmax(dim=-1, keepdim=True)
        selected_offset = top_offsets.gather(1, selected_index).squeeze(1)
        token_ids = selected_offset + self.codec.vocab_size
        no_approval = ~approved.any(dim=1) if require_approval else torch.zeros_like(
            selected_offset, dtype=torch.bool
        )
        if bool(no_approval.any()):
            token_ids = torch.where(
                no_approval, torch.full_like(token_ids, self.codec.wait_id), token_ids
            )
        selected_confidence = F.softmax(base.logits.float(), dim=-1).gather(
            1, selected_offset.unsqueeze(1)
        ).squeeze(1)
        verification = {
            "candidate_token_ids": top_offsets + self.codec.vocab_size,
            "candidate_log_probabilities": top_log_probability,
            "candidate_satisfaction": satisfaction,
            "candidate_votes": votes,
            "candidate_approved": approved,
            "selected_index": selected_index.squeeze(1),
            "min_votes": int(min_votes),
            "expected_postcondition": expected_postcondition,
        }
        return CUADecision(
            token_ids=token_ids,
            actions=self.decode(token_ids.reshape(-1)),
            logits=base.logits,
            confidence=selected_confidence,
            entropy=base.entropy,
            abstained=no_approval,
            needs_confirmation=torch.zeros_like(no_approval),
            verification=verification,
        )

    def action_loss(self, hidden_states: torch.Tensor, action_token_ids: torch.Tensor) -> torch.Tensor:
        """Cross-entropy for CUA trajectories; -100 padding is ignored."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (B, T, D)")
        action_token_ids = action_token_ids.to(device=hidden_states.device, dtype=torch.long)
        if action_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("action_token_ids must have shape (B, T)")
        offsets = action_token_ids - self.codec.vocab_size
        padding = action_token_ids == -100
        valid = (offsets >= 0) & (offsets < self.codec.n_actions)
        if not bool(valid.any()):
            raise ValueError("No supervised action tokens were supplied")
        if bool((~valid & ~padding).any()):
            raise ValueError("Every CUA label must be a reserved action token or -100")
        logits = self.action_head(self._prepare_hidden(hidden_states))
        targets = torch.where(valid, offsets, torch.full_like(offsets, -100))
        return F.cross_entropy(
            logits.reshape(-1, self.codec.n_actions), targets.reshape(-1), ignore_index=-100
        )

    def observe(self, *args, **kwargs) -> CUADecision:
        """Run the owning model and propose an action from its final hidden state."""
        if hasattr(self.model, "hidden_states"):
            hidden = self.model.hidden_states(*args, **kwargs)
        else:
            kwargs["output_hidden_states"] = True
            output = self.model.base_model(*args, **kwargs)
            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is None and isinstance(output, dict):
                hidden_states = output.get("hidden_states")
            if not hidden_states:
                raise ValueError("The adapted base must return hidden states for CUA observe()")
            hidden = hidden_states[-1]
        return self.propose(hidden)

    def observe_if_changed(
        self,
        screen,
        *args,
        detector: ScreenChangeDetector,
        **kwargs,
    ) -> Optional[CUADecision]:
        """Skip model invocation while a bounded screen-diff gate reports no change."""
        if not detector.should_observe(screen):
            return None
        return self.observe(*args, **kwargs)

    def observe_screenshot(self, screenshots, **kwargs) -> CUADecision:
        """Use an adapted model's opt-in encoderless screenshot projector."""
        projector = getattr(self.model, "screenshot_hidden", None)
        if projector is None:
            raise RuntimeError("This model does not expose adapted screenshot input")
        return self.propose(projector(screenshots), **kwargs)

    @staticmethod
    def shadow(driver=None) -> ShadowDriver:
        return ShadowDriver(driver)

    def execute(
        self,
        actions_or_token_ids,
        driver,
        *,
        confirmed: bool = False,
        irreversible: bool = False,
        policy: Optional[CUAPolicy] = None,
        postcondition=None,
        ledger: Optional[CUAReplayLedger] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        audit: bool = True,
    ) -> List[Action]:
        """Validate, guard, execute, verify, and audit a whole action batch."""
        if isinstance(actions_or_token_ids, torch.Tensor) or (
            isinstance(actions_or_token_ids, (list, tuple))
            and (not actions_or_token_ids or isinstance(actions_or_token_ids[0], int))
        ):
            actions = self.decode(actions_or_token_ids)
        else:
            actions = list(actions_or_token_ids)
        policy = policy or getattr(driver, "policy", None)
        book = self.ledger if ledger is None else ledger
        irreversible = bool(
            irreversible
            or any(action.irreversible for action in actions)
            or (policy is not None and policy.requires_confirmation(actions))
        )
        status, error = "executed", None
        try:
            if policy is not None:
                policy.validate(actions)
            if irreversible and not confirmed:
                raise ConfirmationRequired(
                    "Irreversible CUA actions require confirmed=True; no action was executed"
                )
            for action in actions:
                action.execute(driver)
            if postcondition is not None:
                verify = getattr(driver, "verify", None)
                if verify is None:
                    raise PostconditionFailed("The CUA driver cannot verify postconditions")
                if not bool(verify(postcondition)):
                    raise PostconditionFailed(f"CUA postcondition failed: {postcondition!r}")
            if policy is not None:
                policy.mark_executed()
            return actions
        except Exception as exc:
            status, error = "rejected" if isinstance(exc, (ConfirmationRequired, PolicyViolation)) else "failed", str(exc)
            raise
        finally:
            if audit and book is not None:
                book.append(
                    actions,
                    confirmed=confirmed,
                    irreversible=irreversible,
                    driver=type(driver).__name__,
                    status=status,
                    postcondition=postcondition,
                    metadata={**dict(metadata or {}), "model": type(self.model).__name__},
                    error=error,
                )

    def act(
        self,
        hidden_states: torch.Tensor,
        driver,
        *,
        confirmed: bool = False,
        irreversible: bool = False,
        position: int = -1,
        policy: Optional[CUAPolicy] = None,
        postcondition=None,
        shadow: bool = False,
        **propose_kwargs,
    ) -> CUADecision:
        """Propose and execute one CUA step, optionally through a shadow driver."""
        decision = self.propose(hidden_states, position=position, **propose_kwargs)
        if decision.needs_confirmation is not None and bool(decision.needs_confirmation.any()) and not confirmed:
            raise ConfirmationRequired("Uncertain CUA proposal requires confirmed=True")
        driver = ShadowDriver(driver) if shadow else driver
        self.execute(
            decision.actions,
            driver,
            confirmed=confirmed,
            irreversible=irreversible,
            policy=policy,
            postcondition=postcondition,
        )
        return decision

    def act_verified(
        self,
        hidden_states: torch.Tensor,
        driver,
        *,
        confirmed: bool = False,
        irreversible: bool = False,
        shadow: bool = False,
        **verified_kwargs,
    ) -> CUADecision:
        decision = self.propose_verified(hidden_states, **verified_kwargs)
        expected = verified_kwargs.get("expected_postcondition")
        driver = ShadowDriver(driver) if shadow else driver
        self.execute(
            decision.actions,
            driver,
            confirmed=confirmed,
            irreversible=irreversible,
            postcondition=expected,
        )
        return decision


def freeze_for_cua(model) -> None:
    """Freeze the existing model and unfreeze only CUA-trainable components."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    for name in ("screenshot_projector", "cua_adapters"):
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)


DEFAULT_CUA_CANARIES = (
    "clicking",
    "scrolling",
    "typing",
    "interruption_recovery",
    "wrong_screen_detection",
    "irreversible_action_rejection",
)


def run_cua_contract_canaries(models: Mapping[str, object]) -> Dict[str, bool]:
    """Run side-effect-free CUA contract checks on named model instances."""
    results: Dict[str, bool] = {
        "clicking": True,
        "scrolling": True,
        "typing": True,
        "interruption_recovery": True,
        "wrong_screen_detection": True,
        "irreversible_action_rejection": True,
        "policy_rejection": True,
        "postconditions": True,
    }
    detector = ScreenChangeDetector(threshold=0.01)
    results["wrong_screen_detection"] = (
        detector.should_observe(torch.zeros(3, 8, 8))
        and not detector.should_observe(torch.zeros(3, 8, 8))
        and detector.should_observe(torch.ones(3, 8, 8))
    )
    for name, model in models.items():
        agent = model.cua
        dim = agent.action_head.proj.in_features
        hidden = torch.zeros(1, 2, dim)
        labels = torch.full((1, 2), agent.codec.wait_id, dtype=torch.long)
        contract = bool(torch.isfinite(agent.action_loss(hidden, labels)))
        shadow = ShadowDriver()
        agent.execute([
            Action("click", x=0.5, y=0.5),
            Action("scroll", dy=1),
            Action("type", text="safe"),
        ], shadow)
        kinds = [event[0] for event in shadow.events]
        results["clicking"] &= "click" in kinds
        results["scrolling"] &= "scroll" in kinds
        results["typing"] &= "type" in kinds
        blocked = ShadowDriver()
        try:
            agent.execute([Action("type", text="send", irreversible=True)], blocked)
            results["irreversible_action_rejection"] = False
        except ConfirmationRequired:
            pass
        results["irreversible_action_rejection"] &= not blocked.events
        try:
            agent.execute(
                [Action("click", x=2.0, y=0.5)],
                ShadowDriver(),
                policy=CUAPolicy(),
            )
            results["policy_rejection"] = False
        except PolicyViolation:
            pass
        shadow.postconditions["ready"] = True
        try:
            agent.execute([Action("click", x=0.5, y=0.5)], shadow, postcondition="ready")
        except PostconditionFailed:
            results["postconditions"] = False
        trajectory = CUATrajectory(
            hidden,
            torch.tensor([[agent.codec.encode_click(0.5, 0.5), agent.codec.wait_id]]),
            verified=True,
        )
        variants = build_recovery_curriculum(agent.codec, [trajectory])
        results["interruption_recovery"] &= {
            item.metadata["recovery_event"] for item in variants
        } == set(RECOVERY_EVENTS)
        results[f"contract_{name}"] = contract
    return results


def cua_release_gate(
    canaries: Mapping[str, bool],
    *,
    required: Optional[Sequence[str]] = None,
    min_pass_rate: float = 1.0,
) -> dict:
    """Require grounded CUA canaries and all safety-critical checks to pass."""
    required = tuple(required or DEFAULT_CUA_CANARIES)
    missing = [name for name in required if name not in canaries]
    failed = [name for name in required if name in canaries and not canaries[name]]
    values = [bool(value) for value in canaries.values()]
    pass_rate = sum(values) / max(1, len(values))
    safety_failed = [
        name for name in ("irreversible_action_rejection", "wrong_screen_detection")
        if not canaries.get(name, False)
    ]
    passed = not missing and not failed and not safety_failed and pass_rate >= min_pass_rate
    return {
        "passed": bool(passed),
        "pass_rate": float(pass_rate),
        "missing": missing,
        "failed": failed,
        "safety_failed": safety_failed,
    }


def load_cua_compatible_state(model, state) -> None:
    """Load native checkpoints, including pre-universal-CUA action-head absence.

    Only the newly introduced action-head weight may be absent. Any other
    missing/unexpected key remains a hard checkpoint error. Legacy checkpoints
    receive freshly initialized CUA weights and explicit restoration metadata.
    """
    result = model.load_state_dict(state, strict=False)
    action_keys = {"action_head.proj.weight"}
    missing = set(result.missing_keys)
    unexpected = set(result.unexpected_keys)
    if unexpected or missing - action_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.cua.restored_from_legacy = bool(missing & action_keys)
