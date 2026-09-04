import pytest
import torch
import torch.nn.functional as functional

from scaling_lm.config import MODEL_SIZES
from scaling_lm.model import GPT, CausalSelfAttention, GPTConfig, LayerNorm, causal_mask

BATCH = 2
SEQ = 8
SMALL_CONFIG = GPTConfig(n_layer=2, d_model=32, n_head=4, vocab_size=64, context_length=16)


def test_attention_output_shape_matches_input():
    attention = CausalSelfAttention(SMALL_CONFIG)
    hidden = torch.randn(BATCH, SEQ, SMALL_CONFIG.d_model)
    assert attention(hidden).shape == (BATCH, SEQ, SMALL_CONFIG.d_model)


def test_attention_weights_are_causal_and_normalised():
    attention = CausalSelfAttention(SMALL_CONFIG)
    queries = torch.randn(BATCH, SMALL_CONFIG.n_head, SEQ, SMALL_CONFIG.head_dim)
    keys = torch.randn(BATCH, SMALL_CONFIG.n_head, SEQ, SMALL_CONFIG.head_dim)
    weights = attention.attention_weights(queries, keys)
    future = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[..., future] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH, SMALL_CONFIG.n_head, SEQ))


def test_attention_matches_torch_reference():
    torch.manual_seed(0)
    attention = CausalSelfAttention(SMALL_CONFIG)
    hidden = torch.randn(BATCH, SEQ, SMALL_CONFIG.d_model)
    queries, keys, values = attention.qkv_projection(hidden).split(SMALL_CONFIG.d_model, dim=-1)
    queries, keys, values = map(attention.split_heads, (queries, keys, values))
    reference = functional.scaled_dot_product_attention(queries, keys, values, is_causal=True)
    reference = reference.transpose(1, 2).reshape(BATCH, SEQ, SMALL_CONFIG.d_model)
    assert torch.allclose(attention(hidden), attention.output_projection(reference), atol=1e-5)


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(4)
    expected = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool
    )
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("scheme", ["learned", "sinusoidal", "rope"])
def test_changing_a_future_token_does_not_change_earlier_logits(scheme):
    torch.manual_seed(0)
    config = GPTConfig(
        n_layer=2, d_model=32, n_head=4, vocab_size=64, context_length=16, positional_scheme=scheme
    )
    model = GPT(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, SEQ))
    altered = tokens.clone()
    altered[0, -1] = (altered[0, -1] + 1) % config.vocab_size
    with torch.no_grad():
        logits, _ = model(tokens)
        altered_logits, _ = model(altered)
    assert torch.allclose(logits[:, :-1], altered_logits[:, :-1], atol=1e-5)
    assert not torch.allclose(logits[:, -1], altered_logits[:, -1], atol=1e-5)


def test_layer_norm_matches_torch():
    layer_norm = LayerNorm(SMALL_CONFIG.d_model)
    hidden = torch.randn(BATCH, SEQ, SMALL_CONFIG.d_model) * 3 + 2
    reference = functional.layer_norm(hidden, (SMALL_CONFIG.d_model,), eps=layer_norm.eps)
    assert torch.allclose(layer_norm(hidden), reference, atol=1e-5)


def test_forward_returns_logits_and_loss():
    model = GPT(SMALL_CONFIG)
    tokens = torch.randint(0, SMALL_CONFIG.vocab_size, (BATCH, SEQ))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (BATCH, SEQ, SMALL_CONFIG.vocab_size)
    assert loss is not None and loss.ndim == 0 and loss.item() > 0


def seeded_state_dict(scheme: str, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    config = GPTConfig(
        n_layer=2, d_model=32, n_head=4, vocab_size=64, context_length=16, positional_scheme=scheme
    )
    return GPT(config).state_dict()


@pytest.mark.parametrize("alternative", ["rope", "sinusoidal"])
def test_same_seed_gives_identical_shared_parameters_across_positional_schemes(alternative):
    """The ablation pairs runs by seed, which only means something if the seed fixes the
    initial weights of everything the two schemes share, not just the data order."""
    learned = seeded_state_dict("learned", seed=3)
    other = seeded_state_dict(alternative, seed=3)
    shared = learned.keys() & other.keys()
    assert shared, "no shared parameters to compare"
    differing = sorted(name for name in shared if not torch.equal(learned[name], other[name]))
    assert differing == [], f"shared parameters differ between schemes: {differing}"
    assert seeded_state_dict("learned", seed=4).keys() == learned.keys()
    assert not torch.equal(
        seeded_state_dict("learned", seed=4)["blocks.0.mlp.expand.weight"],
        learned["blocks.0.mlp.expand.weight"],
    )


def test_learned_position_table_is_initialised_like_the_other_embeddings():
    torch.manual_seed(0)
    model = GPT(SMALL_CONFIG)
    table = model.positional.table
    assert table.requires_grad
    assert table.std().item() == pytest.approx(SMALL_CONFIG.init_std, rel=0.2)


def test_weight_tying_and_parameter_counts():
    model = GPT(SMALL_CONFIG)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    counts = model.count_parameters()
    positional = SMALL_CONFIG.context_length * SMALL_CONFIG.d_model
    token = SMALL_CONFIG.vocab_size * SMALL_CONFIG.d_model
    assert counts["embedding"] == token + positional
    assert counts["total"] == counts["embedding"] + counts["non_embedding"]


@pytest.mark.parametrize("size", MODEL_SIZES, ids=lambda size: size.name)
def test_sweep_sizes_match_kaplan_estimate(size):
    model = GPT(GPTConfig.from_model_size(size, "learned"))
    non_embedding = model.count_parameters()["non_embedding"]
    # Biases and norm gains add a little on top of the 12 * n_layer * d_model^2 estimate.
    assert non_embedding >= size.approx_non_embedding_params
    assert non_embedding < 1.02 * size.approx_non_embedding_params


def test_generate_extends_sequence_within_context():
    model = GPT(SMALL_CONFIG).eval()
    prompt = torch.randint(0, SMALL_CONFIG.vocab_size, (1, 3))
    output = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=8)
    assert output.shape == (1, 8)
    assert torch.equal(output[:, :3], prompt)
