"""Masked candidate-scoring actor/critic used by the Stage 14 curriculum."""

from __future__ import annotations

# NumPy must be imported before torch in the current Apple Silicon ROS/Pixi
# environment so both libraries bind the same OpenMP runtime.
import numpy as np
import torch
from torch import nn

from .stage12_encoding import Stage12Encoder


class MaskedCandidateActorCritic(nn.Module):
    """Score each dynamic assignment from shared state and candidate features.

    No trainable parameter is tied to a particular action index. The same
    candidate encoder is applied to every slot, which prevents the policy from
    memorizing that a transient assignment happened to occupy index zero.
    """

    def __init__(self, state_width: int = Stage12Encoder.OBSERVATION_SIZE,
                 action_width: int = Stage12Encoder.ACTION_WIDTH,
                 hidden_width: int = 96,
                 policy_variant: str = "CONTEXT_INTERACTION"):
        super().__init__()
        policy_variant = policy_variant.upper()
        if policy_variant not in {"CONTEXT_INTERACTION", "FLAT_MASKED_PPO"}:
            raise ValueError(f"unsupported policy variant: {policy_variant}")
        self.state_width = state_width
        self.action_width = action_width
        self.hidden_width = hidden_width
        self.policy_variant = policy_variant
        self.state_encoder = nn.Sequential(
            nn.Linear(state_width, hidden_width), nn.Tanh(),
            nn.Linear(hidden_width, hidden_width), nn.Tanh())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_width, hidden_width), nn.Tanh(),
            nn.Linear(hidden_width, hidden_width))
        # Explicit state/candidate interaction is important here: a candidate
        # is good only relative to the current task geometry, robot positions,
        # reservations and deadlines.  Concatenating both embeddings and their
        # elementwise product prevents the actor from reducing the decision to
        # a state-independent global preference for one transport mode.
        actor_input_width = hidden_width * (
            3 if policy_variant == "CONTEXT_INTERACTION" else 2)
        self.actor_head = nn.Sequential(
            nn.Linear(actor_input_width, hidden_width), nn.Tanh(),
            nn.Linear(hidden_width, 1))
        self.value_head = nn.Sequential(
            nn.Linear(hidden_width, hidden_width), nn.Tanh(),
            nn.Linear(hidden_width, 1))

    def forward(self, state: torch.Tensor, action_features: torch.Tensor,
                action_mask: torch.Tensor):
        unbatched = state.ndim == 1
        if unbatched:
            state = state.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        if state.ndim != 2 or action_features.ndim != 3:
            raise ValueError("state/action tensors have invalid rank")
        if not torch.all(action_mask.any(dim=1)):
            raise ValueError("every policy sample needs a legal action")
        state_embedding = self.state_encoder(state)
        candidate_embedding = self.action_encoder(action_features)
        expanded_state = state_embedding.unsqueeze(1).expand_as(
            candidate_embedding)
        if self.policy_variant == "CONTEXT_INTERACTION":
            joint = torch.cat((
                candidate_embedding, expanded_state,
                candidate_embedding * expanded_state), dim=-1)
        else:
            joint = torch.cat((candidate_embedding, expanded_state), dim=-1)
        logits = self.actor_head(joint).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(),
                                    torch.finfo(logits.dtype).min)
        value = self.value_head(state_embedding).squeeze(-1)
        if unbatched:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    @torch.no_grad()
    def select_action(self, observation: dict[str, np.ndarray],
                      action_mask: np.ndarray,
                      deterministic: bool = True) -> int:
        state = torch.as_tensor(observation["state"], dtype=torch.float32)
        features = torch.as_tensor(
            observation["action_features"][..., :self.action_width],
            dtype=torch.float32)
        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        logits, _ = self(state, features, mask)
        if deterministic:
            return int(torch.argmax(logits).item())
        distribution = torch.distributions.Categorical(logits=logits)
        return int(distribution.sample().item())

    def metadata(self) -> dict:
        return {
            "architecture": (
                "contextual_masked_candidate_actor_critic_v2" if
                self.policy_variant == "CONTEXT_INTERACTION" else
                "flat_masked_candidate_actor_critic_v1"),
            "policy_variant": self.policy_variant,
            "state_width": self.state_width,
            "action_width": self.action_width,
            "hidden_width": self.hidden_width,
            "index_specific_parameters": False,
        }
