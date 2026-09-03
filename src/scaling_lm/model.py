"""Decoder-only transformer written from the tensor operations up.

Only nn.Linear, nn.Embedding and the functional GELU / softmax / cross entropy
primitives are borrowed from PyTorch. Layer normalisation, attention, the
causal mask and the positional encodings are all implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from scaling_lm.config import (
    CONTEXT_LENGTH,
    DEFAULT_POSITIONAL_SCHEME,
    FREQUENCY_BASE,
    HEAD_DIM,
    INIT_STD,
    LAYER_NORM_EPS,
    MLP_EXPANSION,
    POSITIONAL_SCHEMES,
    VOCAB_SIZE,
    ModelSize,
)
from scaling_lm.positional import (
    LearnedPositionalEmbedding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from scaling_lm.validation import require_positive


@dataclass(frozen=True)
class GPTConfig:
    """Full architecture specification for one model.

    The model code reads every numeric setting from here and nowhere else, so the
    resolved config is the complete architecture record that `runs.py` fingerprints.
    """

    n_layer: int
    d_model: int
    n_head: int
    vocab_size: int = VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH
    mlp_expansion: int = MLP_EXPANSION
    positional_scheme: str = DEFAULT_POSITIONAL_SCHEME
    init_std: float = INIT_STD
    layer_norm_eps: float = LAYER_NORM_EPS
    positional_frequency_base: float = FREQUENCY_BASE

    def __post_init__(self) -> None:
        require_positive("n_layer", self.n_layer)
        require_positive("d_model", self.d_model)
        require_positive("n_head", self.n_head)
        require_positive("vocab_size", self.vocab_size)
        require_positive("context_length", self.context_length)
        require_positive("mlp_expansion", self.mlp_expansion)
        require_positive("init_std", self.init_std)
        require_positive("layer_norm_eps", self.layer_norm_eps)
        require_positive("positional_frequency_base", self.positional_frequency_base)
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.positional_scheme not in POSITIONAL_SCHEMES:
            raise ValueError(f"unknown positional scheme: {self.positional_scheme}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @classmethod
    def from_model_size(cls, size: ModelSize, positional_scheme: str) -> GPTConfig:
        return cls(
            n_layer=size.n_layer,
            d_model=size.d_model,
            n_head=size.d_model // HEAD_DIM,
            positional_scheme=positional_scheme,
        )


class LayerNorm(nn.Module):
    """Normalise each token's feature vector to zero mean and unit variance, then rescale."""

    def __init__(self, d_model: int, eps: float = LAYER_NORM_EPS) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = hidden.var(dim=-1, keepdim=True, unbiased=False)
        normalised = (hidden - mean) * torch.rsqrt(variance + self.eps)
        return normalised * self.gain + self.bias


def causal_mask(sequence_length: int, device: torch.device | None = None) -> Tensor:
    """Boolean (seq, seq) mask that is True where a query may attend to a key (key <= query)."""
    return torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position only sees itself and earlier positions."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.qkv_projection = nn.Linear(config.d_model, 3 * config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.rotary: RotaryPositionalEncoding | None = None
        if config.positional_scheme == "rope":
            self.rotary = RotaryPositionalEncoding(
                config.head_dim, config.context_length, config.positional_frequency_base
            )
        self.register_buffer("mask", causal_mask(config.context_length), persistent=False)

    def split_heads(self, projected: Tensor) -> Tensor:
        """(batch, seq, d_model) -> (batch, heads, seq, head_dim)."""
        batch_size, sequence_length, _ = projected.shape
        return projected.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(
            1, 2
        )

    def forward(self, hidden: Tensor) -> Tensor:
        batch_size, sequence_length, d_model = hidden.shape
        queries, keys, values = self.qkv_projection(hidden).split(d_model, dim=-1)
        queries, keys, values = map(self.split_heads, (queries, keys, values))
        if self.rotary is not None:
            queries, keys = self.rotary(queries, keys)

        attention_weights = self.attention_weights(queries, keys)
        context = attention_weights @ values
        merged = context.transpose(1, 2).reshape(batch_size, sequence_length, d_model)
        return self.output_projection(merged)

    def attention_weights(self, queries: Tensor, keys: Tensor) -> Tensor:
        """Masked, scaled softmax(QK^T) of shape (batch, heads, seq, seq)."""
        sequence_length = queries.shape[-2]
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = self.mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~allowed, float("-inf"))
        return functional.softmax(scores, dim=-1)


class FeedForward(nn.Module):
    """Position-wise two-layer MLP with a GELU non-linearity."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_expansion * config.d_model
        self.expand = nn.Linear(config.d_model, hidden_dim)
        self.contract = nn.Linear(hidden_dim, config.d_model)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.contract(functional.gelu(self.expand(hidden)))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(config.d_model, config.layer_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = LayerNorm(config.d_model, config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


def build_additive_positional(config: GPTConfig) -> nn.Module | None:
    """Return the input-side positional module, or None when RoPE handles positions."""
    if config.positional_scheme == "learned":
        return LearnedPositionalEmbedding(config.context_length, config.d_model)
    if config.positional_scheme == "sinusoidal":
        return SinusoidalPositionalEncoding(
            config.context_length, config.d_model, config.positional_frequency_base
        )
    return None


class GPT(nn.Module):
    """GPT-style causal language model with tied input and output embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional = build_additive_positional(config)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.final_norm = LayerNorm(config.d_model, config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying halves the embedding parameter count and is standard for GPT-2.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def _scale_residual_projections(self) -> None:
        # Each block adds two residual contributions, so shrink their output
        # projections by 1/sqrt(2 * n_layer) to keep the residual stream variance
        # roughly constant with depth (Radford et al., 2019).
        scale = self.config.init_std / math.sqrt(2 * self.config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attention.output_projection.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.contract.weight, mean=0.0, std=scale)

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return logits (batch, seq, vocab) and, when targets are given, the mean cross entropy."""
        sequence_length = input_ids.shape[1]
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context {self.config.context_length}"
            )
        hidden = self.token_embedding(input_ids)
        if self.positional is not None:
            hidden = hidden + self.positional(sequence_length)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if targets is not None:
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss

    def count_parameters(self) -> dict[str, int]:
        """Total and non-embedding parameter counts (Kaplan et al. report the latter)."""
        total = sum(parameter.numel() for parameter in self.parameters())
        embedding = self.token_embedding.weight.numel()
        if isinstance(self.positional, LearnedPositionalEmbedding):
            embedding += self.positional.embedding.weight.numel()
        return {"total": total, "embedding": embedding, "non_embedding": total - embedding}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> Tensor:
        """Sample autoregressively, keeping only the last context_length tokens as input."""
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth_largest = torch.topk(next_logits, min(top_k, next_logits.shape[-1])).values
                next_logits[next_logits < kth_largest[:, [-1]]] = float("-inf")
            probabilities = functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
