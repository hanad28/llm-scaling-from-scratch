import math

import torch

from scaling_lm.config import FREQUENCY_BASE
from scaling_lm.positional import (
    LearnedPositionalEmbedding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
    sinusoidal_table,
)

CONTEXT = 32
D_MODEL = 16
HEAD_DIM = 8


def test_sinusoidal_table_matches_closed_form():
    table = sinusoidal_table(CONTEXT, D_MODEL, FREQUENCY_BASE)
    assert table.shape == (CONTEXT, D_MODEL)
    position, pair = 5, 3
    angle = position / FREQUENCY_BASE ** (2 * pair / D_MODEL)
    assert math.isclose(table[position, 2 * pair].item(), math.sin(angle), abs_tol=1e-6)
    assert math.isclose(table[position, 2 * pair + 1].item(), math.cos(angle), abs_tol=1e-6)
    assert torch.allclose(table[0, 0::2], torch.zeros(D_MODEL // 2))
    assert torch.allclose(table[0, 1::2], torch.ones(D_MODEL // 2))


def test_sinusoidal_module_slices_to_sequence_length_and_has_no_parameters():
    encoding = SinusoidalPositionalEncoding(CONTEXT, D_MODEL, FREQUENCY_BASE)
    assert encoding(10).shape == (10, D_MODEL)
    assert sum(parameter.numel() for parameter in encoding.parameters()) == 0


def test_learned_embedding_returns_one_vector_per_position():
    embedding = LearnedPositionalEmbedding(CONTEXT, D_MODEL)
    torch.nn.init.normal_(embedding.table)
    output = embedding(7)
    assert output.shape == (7, D_MODEL)
    assert torch.equal(output[3], embedding.table[3])
    assert embedding.table.requires_grad


def test_rotary_preserves_vector_norms():
    rotary = RotaryPositionalEncoding(HEAD_DIM, CONTEXT, FREQUENCY_BASE)
    queries = torch.randn(2, 3, CONTEXT, HEAD_DIM)
    keys = torch.randn(2, 3, CONTEXT, HEAD_DIM)
    rotated_queries, rotated_keys = rotary(queries, keys)
    assert torch.allclose(rotated_queries.norm(dim=-1), queries.norm(dim=-1), atol=1e-5)
    assert torch.allclose(rotated_keys.norm(dim=-1), keys.norm(dim=-1), atol=1e-5)


def test_rotary_leaves_position_zero_unchanged():
    rotary = RotaryPositionalEncoding(HEAD_DIM, CONTEXT, FREQUENCY_BASE)
    queries = torch.randn(1, 1, CONTEXT, HEAD_DIM)
    rotated, _ = rotary(queries, queries)
    assert torch.allclose(rotated[..., 0, :], queries[..., 0, :], atol=1e-6)


def test_rotary_scores_depend_only_on_relative_position():
    """q_m . k_n must equal q_(m+s) . k_(n+s) for the same underlying vectors."""
    rotary = RotaryPositionalEncoding(HEAD_DIM, CONTEXT, FREQUENCY_BASE)
    torch.manual_seed(0)
    query_vector = torch.randn(HEAD_DIM)
    key_vector = torch.randn(HEAD_DIM)
    queries = query_vector.expand(1, 1, CONTEXT, HEAD_DIM)
    keys = key_vector.expand(1, 1, CONTEXT, HEAD_DIM)
    rotated_queries, rotated_keys = rotary(queries, keys)
    scores = rotated_queries[0, 0] @ rotated_keys[0, 0].T
    for shift in (1, 5, 11):
        shifted = scores[10 + shift, 4 + shift].item()
        assert math.isclose(scores[10, 4].item(), shifted, abs_tol=1e-4)
    assert not math.isclose(scores[10, 4].item(), scores[10, 6].item(), abs_tol=1e-3)
