from __future__ import annotations

import torch
from torch import nn


def stable_slerp_carry(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    interpolation: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Global product-space SLERP, rescaled to the previous state's norm."""
    if previous.shape != candidate.shape:
        raise ValueError("previous and candidate states must have identical shapes")
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError("interpolation must be in [0, 1]")

    previous_norm = torch.linalg.vector_norm(previous)
    candidate_norm = torch.linalg.vector_norm(candidate)
    if previous_norm.detach().item() <= epsilon:
        return candidate
    if candidate_norm.detach().item() <= epsilon:
        return previous

    u = previous / previous_norm.clamp_min(epsilon)
    v = candidate / candidate_norm.clamp_min(epsilon)
    cosine = torch.vdot(u.flatten(), v.flatten()).real.clamp(-1.0, 1.0)
    orthogonal = v - cosine * u
    orthogonal_norm = torch.linalg.vector_norm(orthogonal)
    if orthogonal_norm.detach().item() <= epsilon:
        return previous
    tangent = orthogonal / orthogonal_norm
    accepted_angle = interpolation * torch.acos(cosine)
    direction = torch.cos(accepted_angle) * u + torch.sin(accepted_angle) * tangent
    return direction * previous_norm


class LowRankStateCarrier(nn.Module):
    """Layerwise low-rank key/value reparameterization of GatedDeltaNet states."""

    def __init__(self, layers: int, heads: int, key_dim: int, value_dim: int, rank: int):
        super().__init__()
        if min(layers, heads, key_dim, value_dim, rank) < 1:
            raise ValueError("all dimensions must be positive")
        self.layers = layers
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.rank = rank

        self.key_a = nn.Parameter(torch.empty(layers, rank, key_dim))
        self.key_b = nn.Parameter(torch.zeros(layers, key_dim, rank))
        self.value_b = nn.Parameter(torch.empty(layers, value_dim, rank))
        self.value_a = nn.Parameter(torch.zeros(layers, rank, value_dim))
        self.gate_identity = nn.Parameter(torch.ones(layers, heads))
        self.gate_key = nn.Parameter(torch.ones(layers, heads))
        self.gate_value = nn.Parameter(torch.ones(layers, heads))
        nn.init.normal_(self.key_a, std=0.02)
        nn.init.normal_(self.value_b, std=0.02)

    def forward(self, terminal_state: torch.Tensor) -> torch.Tensor:
        expected = (self.layers, self.heads, self.key_dim, self.value_dim)
        if tuple(terminal_state.shape) != expected:
            raise ValueError(f"expected recurrent state {expected}, got {tuple(terminal_state.shape)}")
        state = terminal_state.float()
        key_projection = torch.einsum("lrk,lhkv->lhrv", self.key_a, state)
        key_update = torch.einsum("lkr,lhrv->lhkv", self.key_b, key_projection)
        value_projection = torch.einsum("lhkv,lvr->lhkr", state, self.value_b)
        value_update = torch.einsum("lhkr,lrv->lhkv", value_projection, self.value_a)
        return (
            self.gate_identity[..., None, None] * state
            + self.gate_key[..., None, None] * key_update
            + self.gate_value[..., None, None] * value_update
        )
