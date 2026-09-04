"""Recurrent candidate-scoring policy for causal online dispatch.

The feed-forward Stage 25 policy receives a causal but partially observed
state.  This module adds an episode-scoped GRU memory while preserving shared
candidate scoring and action masking.  No parameter is tied to an action slot.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .stage25_causal_observation import CausalReleasedEncoder


class RecurrentMaskedCandidateActorCritic(nn.Module):
    """GRU-memory actor critic with shared dynamic-candidate parameters."""

    def __init__(self,
                 state_width: int =
                 CausalReleasedEncoder.MARKOV_OBSERVATION_SIZE,
                 action_width: int = CausalReleasedEncoder.ACTION_WIDTH,
                 hidden_width: int = 96,
                 memory_width: int = 96):
        super().__init__()
        self.state_width = int(state_width)
        self.action_width = int(action_width)
        self.hidden_width = int(hidden_width)
        self.memory_width = int(memory_width)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_width, self.hidden_width), nn.Tanh(),
            nn.Linear(self.hidden_width, self.hidden_width), nn.Tanh())
        self.memory_cell = nn.GRUCell(self.hidden_width, self.memory_width)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_width, self.memory_width), nn.Tanh(),
            nn.Linear(self.memory_width, self.memory_width))
        self.actor_head = nn.Sequential(
            nn.Linear(self.memory_width * 3, self.hidden_width), nn.Tanh(),
            nn.Linear(self.hidden_width, 1))
        self.value_head = nn.Sequential(
            nn.Linear(self.memory_width, self.hidden_width), nn.Tanh(),
            nn.Linear(self.hidden_width, 1))

    def initial_memory(self, batch_size: int = 1, *, device=None,
                       dtype=torch.float32) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            int(batch_size), self.memory_width,
            device=device or parameter.device, dtype=dtype)

    def forward(self, state: torch.Tensor, action_features: torch.Tensor,
                action_mask: torch.Tensor,
                memory: torch.Tensor | None = None):
        unbatched = state.ndim == 1
        if unbatched:
            state = state.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        if state.ndim != 2 or action_features.ndim != 3:
            raise ValueError("state/action tensors have invalid rank")
        if state.shape[-1] != self.state_width:
            raise ValueError(
                f"state width mismatch: {state.shape[-1]} != "
                f"{self.state_width}")
        if action_features.shape[-1] != self.action_width:
            raise ValueError("candidate feature width mismatch")
        if not torch.all(action_mask.any(dim=1)):
            raise ValueError("every policy sample needs a legal action")
        if memory is None:
            memory = self.initial_memory(
                state.shape[0], device=state.device, dtype=state.dtype)
        elif memory.ndim == 1:
            memory = memory.unsqueeze(0)
        if memory.shape != (state.shape[0], self.memory_width):
            raise ValueError("recurrent memory shape mismatch")

        state_embedding = self.state_encoder(state)
        next_memory = self.memory_cell(state_embedding, memory)
        candidate_embedding = self.action_encoder(action_features)
        expanded_memory = next_memory.unsqueeze(1).expand_as(
            candidate_embedding)
        joint = torch.cat((
            candidate_embedding, expanded_memory,
            candidate_embedding * expanded_memory), dim=-1)
        logits = self.actor_head(joint).squeeze(-1)
        logits = logits.masked_fill(
            ~action_mask.bool(), torch.finfo(logits.dtype).min)
        value = self.value_head(next_memory).squeeze(-1)
        if unbatched:
            return (logits.squeeze(0), value.squeeze(0),
                    next_memory.squeeze(0))
        return logits, value, next_memory

    @torch.no_grad()
    def select_action(self, observation: dict[str, np.ndarray],
                      action_mask: np.ndarray,
                      memory: torch.Tensor | None = None,
                      deterministic: bool = True):
        state = torch.as_tensor(observation["state"], dtype=torch.float32)
        features = torch.as_tensor(
            observation["action_features"][..., :self.action_width],
            dtype=torch.float32)
        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        logits, _, next_memory = self(state, features, mask, memory)
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            action = int(
                torch.distributions.Categorical(logits=logits).sample().item())
        return action, next_memory

    def metadata(self) -> dict:
        return {
            "architecture": "recurrent_contextual_masked_candidate_actor_critic_v1",
            "policy_variant": "RECURRENT_CONTEXT_INTERACTION",
            "recurrent_cell": "GRUCell",
            "state_width": self.state_width,
            "action_width": self.action_width,
            "hidden_width": self.hidden_width,
            "memory_width": self.memory_width,
            "memory_reset_boundary": "EPISODE",
            "index_specific_parameters": False,
        }
