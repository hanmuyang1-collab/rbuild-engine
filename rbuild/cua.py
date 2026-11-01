"""Universal computer-using-agent (CUA) contract for every R-Build model.

The same native ActionCodec/ActuationHead action space is used by the text
model, R-OutGen, and adapted pretrained models. A model proposes actions from
its own hidden states; a caller-provided driver performs real OS operations.
Action heads still require computer-use training before they can be competent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from .actuation import Action, ActionCodec, ActuationHead


class ConfirmationRequired(RuntimeError):
    """Raised before any side effect when irreversible work is unconfirmed."""


@dataclass
class CUADecision:
    """Native action-token ids, decoded actions, and the selected logits."""

    token_ids: torch.Tensor
    actions: List[Action]
    logits: torch.Tensor


class ComputerUseAgent:
    """Shared observe/propose/decode/train/execute interface for all models."""

    def __init__(self, model, codec: ActionCodec, action_head: ActuationHead):
        if codec is None or action_head is None:
            raise ValueError("Every R-Build model must provide a CUA codec and action head")
        self.model = model
        self.codec = codec
        self.action_head = action_head
        self.restored_from_legacy = False

    def decode(self, token_ids) -> List[Action]:
        """Decode reserved action tokens; ordinary text tokens are ignored."""
        return self.codec.decode_actions(token_ids)

    @torch.no_grad()
    def propose(self, hidden_states: torch.Tensor, position: int = -1) -> CUADecision:
        """Greedily propose native actions at one hidden-state position."""
        if hidden_states.ndim not in (2, 3):
            raise ValueError("hidden_states must have shape (B, D) or (B, T, D)")
        if hidden_states.shape[-1] != self.action_head.proj.in_features:
            raise ValueError("hidden_states has the wrong final dimension")
        logits = self.action_head(hidden_states)
        selected = logits[:, position] if logits.ndim == 3 else logits
        token_ids = selected.argmax(dim=-1) + self.codec.vocab_size
        actions = self.decode(token_ids.reshape(-1))
        return CUADecision(token_ids=token_ids, actions=actions, logits=selected)

    def action_loss(self, hidden_states: torch.Tensor, action_token_ids: torch.Tensor) -> torch.Tensor:
        """Cross-entropy for supervised CUA trajectories using reserved token ids."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (B, T, D)")
        action_token_ids = action_token_ids.to(device=hidden_states.device, dtype=torch.long)
        if action_token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("action_token_ids must have shape (B, T)")
        offsets = action_token_ids - self.codec.vocab_size
        valid = (offsets >= 0) & (offsets < self.codec.n_actions)
        if not bool(valid.any()):
            raise ValueError("No supervised action tokens were supplied")
        if not bool(valid.all()):
            raise ValueError("Every CUA label must be a reserved action token")
        logits = self.action_head(hidden_states)
        return F.cross_entropy(logits.reshape(-1, self.codec.n_actions), offsets.reshape(-1))

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

    def execute(
        self,
        actions_or_token_ids,
        driver,
        *,
        confirmed: bool = False,
        irreversible: bool = False,
    ) -> List[Action]:
        """Execute through a real driver, guarding the whole batch atomically.

        Set ``irreversible=True`` when the batch may cause an irreversible side
        effect. Individual Action objects can also set ``irreversible=True``.
        Confirmation is checked before the first action, never mid-stream.
        """
        if isinstance(actions_or_token_ids, torch.Tensor) or (
            isinstance(actions_or_token_ids, (list, tuple))
            and (not actions_or_token_ids or isinstance(actions_or_token_ids[0], int))
        ):
            actions = self.decode(actions_or_token_ids)
        else:
            actions = list(actions_or_token_ids)
        if (irreversible or any(action.irreversible for action in actions)) and not confirmed:
            raise ConfirmationRequired(
                "Irreversible CUA actions require confirmed=True; no action was executed"
            )
        for action in actions:
            action.execute(driver)
        return actions

    def act(
        self,
        hidden_states: torch.Tensor,
        driver,
        *,
        confirmed: bool = False,
        irreversible: bool = False,
        position: int = -1,
    ) -> CUADecision:
        """Propose and execute one CUA step through the supplied real driver."""
        decision = self.propose(hidden_states, position=position)
        self.execute(
            decision.actions,
            driver,
            confirmed=confirmed,
            irreversible=irreversible,
        )
        return decision


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
