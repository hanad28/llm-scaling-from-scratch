"""generate.py records each run's samples against the fingerprint that produced them."""

from __future__ import annotations

import json

import pytest
from conftest import SMOKE_TRAINING

from scaling_lm import generate as generate_module
from scaling_lm.config import ResultsPaths, RunConfig
from scaling_lm.generate import generate_for_runs
from scaling_lm.runs import load_run
from scaling_lm.train import train_run


@pytest.fixture(autouse=True)
def stub_continuation(monkeypatch):
    """Avoid depending on the real tokenizer: a fixed continuation per prompt is enough here."""
    monkeypatch.setattr(
        generate_module,
        "generate_continuation",
        lambda model, prompt, device: f"...{prompt}",
    )


def test_generate_for_runs_records_the_run_fingerprint_with_its_samples(tmp_path, synthetic_corpus):
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", training=SMOKE_TRAINING)
    train_run(run_config, paths)

    generations = generate_for_runs([run_config.run_name], paths)

    expected_fingerprint = load_run(run_config.run_name, paths).identity.fingerprint
    entry = generations[run_config.run_name]
    assert entry["fingerprint"] == expected_fingerprint
    assert set(entry["samples"]) == set(generate_module.GENERATION_PROMPTS)

    on_disk = json.loads(paths.generations.read_text())
    assert on_disk == generations
