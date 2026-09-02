import numpy as np
import pytest
import torch

from scaling_lm import dataset as dataset_module
from scaling_lm.config import SPLIT_FRACTIONS, SPLIT_NAMES
from scaling_lm.data import (
    Document,
    assign_split,
    build_document_text,
    deduplicate,
    has_target_category,
    split_documents,
)
from scaling_lm.dataset import TokenWindows, epoch_batches, sequential_batches
from scaling_lm.train import learning_rate_at


def test_has_target_category():
    assert has_target_category("cs.LG stat.ML")
    assert has_target_category("math.OC cs.AI")
    assert not has_target_category("math.OC stat.ML")
    assert not has_target_category("")


def test_build_document_text_normalises_whitespace():
    text = build_document_text("  A   title\n", "Line one\n  line two.  ")
    assert text == "A title\nLine one line two."


def test_assign_split_is_deterministic_and_roughly_proportional():
    ids = [f"2301.{index:05d}" for index in range(20_000)]
    splits = [assign_split(arxiv_id) for arxiv_id in ids]
    assert splits == [assign_split(arxiv_id) for arxiv_id in ids]
    for split_name in SPLIT_NAMES:
        fraction = splits.count(split_name) / len(ids)
        assert fraction == pytest.approx(SPLIT_FRACTIONS[split_name], abs=0.005)


def test_deduplicate_drops_repeated_text():
    documents = [
        Document("a", "cs.LG", "same text"),
        Document("b", "cs.LG", "same text"),
        Document("c", "cs.LG", "other text"),
    ]
    assert [doc.arxiv_id for doc in deduplicate(iter(documents))] == ["a", "c"]


def test_split_documents_has_no_overlap():
    documents = [Document(f"id{index}", "cs.AI", f"text {index}") for index in range(5_000)]
    splits = split_documents(documents)
    ids_per_split = [{doc.arxiv_id for doc in docs} for docs in splits.values()]
    assert sum(len(ids) for ids in ids_per_split) == len(documents)
    assert not (ids_per_split[0] & ids_per_split[1])
    assert not (ids_per_split[0] & ids_per_split[2])
    assert not (ids_per_split[1] & ids_per_split[2])


@pytest.fixture
def fake_tokens(monkeypatch, tmp_path):
    tokens = np.arange(1, 1_001, dtype="uint16")
    path = tmp_path / "train.bin"
    tokens.tofile(path)
    memmap = np.memmap(path, dtype="uint16", mode="r")
    monkeypatch.setattr(dataset_module, "load_tokens", lambda split_name: memmap)
    return tokens


def test_token_windows_targets_are_shifted_inputs(fake_tokens):
    windows = TokenWindows("train", context_length=10)
    assert len(windows) == 99
    inputs, targets = windows.batch(np.array([0, 3]), torch.device("cpu"))
    assert inputs.shape == targets.shape == (2, 10)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert inputs[1, 0].item() == fake_tokens[30]


def test_epoch_batches_cover_each_window_once(fake_tokens):
    windows = TokenWindows("train", context_length=10)
    batches = list(epoch_batches(windows, batch_size=8, seed=0))
    seen = np.concatenate(batches)
    assert len(batches) == 99 // 8
    assert len(np.unique(seen)) == len(seen)
    assert set(np.concatenate(list(sequential_batches(windows, 8)))) == set(range(99))


def test_epoch_order_depends_on_seed_but_not_on_call(fake_tokens):
    windows = TokenWindows("train", context_length=10)
    first = np.concatenate(list(epoch_batches(windows, 8, seed=0)))
    again = np.concatenate(list(epoch_batches(windows, 8, seed=0)))
    other = np.concatenate(list(epoch_batches(windows, 8, seed=1)))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_learning_rate_schedule_shape():
    total_steps, peak = 1_000, 1e-3
    rates = [learning_rate_at(step, total_steps, peak) for step in range(total_steps)]
    warmup_end = int(0.05 * total_steps) - 1
    assert rates[warmup_end] == pytest.approx(peak)
    assert max(rates) == pytest.approx(peak)
    assert rates[-1] == pytest.approx(0.1 * peak, rel=1e-3)
    decay = rates[warmup_end:]
    assert all(later <= earlier for earlier, later in zip(decay, decay[1:], strict=False))
