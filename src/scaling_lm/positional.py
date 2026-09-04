"""Positional encoding schemes: learned, sinusoidal and rotary (RoPE).

Learned and sinusoidal encodings are added to the token embeddings once at the
input. RoPE instead rotates the query and key vectors inside every attention
layer, so it is exposed as a separate module that the attention block calls.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_table(context_length: int, d_model: int, frequency_base: float) -> Tensor:
    """Return the fixed (context_length, d_model) sine/cosine table of Vaswani et al. (2017).

    Even columns hold sin(pos / base^(2i/d)), odd columns hold the matching cosine.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal encoding")
    positions = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
    pair_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    inverse_frequencies = torch.exp(-math.log(frequency_base) * pair_indices / d_model)
    angles = positions * inverse_frequencies
    table = torch.zeros(context_length, d_model)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table


class LearnedPositionalEmbedding(nn.Module):
    """One trainable vector per position, as in GPT-2.

    The table is allocated empty and initialised by `GPT`, after every other parameter,
    so constructing it draws nothing from the random number generator.
    """

    def __init__(self, context_length: int, d_model: int) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.empty(context_length, d_model))

    def forward(self, sequence_length: int) -> Tensor:
        return self.table[:sequence_length]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine encoding with no trainable parameters."""

    def __init__(self, context_length: int, d_model: int, frequency_base: float) -> None:
        super().__init__()
        table = sinusoidal_table(context_length, d_model, frequency_base)
        self.register_buffer("table", table, persistent=False)

    def forward(self, sequence_length: int) -> Tensor:
        return self.table[:sequence_length]


class RotaryPositionalEncoding(nn.Module):
    """Rotary position embedding (Su et al., 2024).

    Each pair of channels in a head is treated as a 2D vector and rotated by an
    angle proportional to the token position. Because rotation preserves dot
    products up to the angle difference, q_m . k_n depends only on m - n.
    """

    def __init__(self, head_dim: int, context_length: int, frequency_base: float) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary encoding")
        pair_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inverse_frequencies = 1.0 / (frequency_base ** (pair_indices / head_dim))
        positions = torch.arange(context_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cos_table", torch.cos(angles), persistent=False)
        self.register_buffer("sin_table", torch.sin(angles), persistent=False)

    def forward(self, queries: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate (batch, heads, seq, head_dim) query and key tensors in place of position ids."""
        sequence_length = queries.shape[-2]
        cos = self.cos_table[:sequence_length].to(queries.dtype)
        sin = self.sin_table[:sequence_length].to(queries.dtype)
        return _rotate_pairs(queries, cos, sin), _rotate_pairs(keys, cos, sin)


def _rotate_pairs(vectors: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply a 2D rotation to consecutive channel pairs (x_2i, x_2i+1)."""
    first, second = vectors[..., 0::2], vectors[..., 1::2]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
