"""An interrupted save must never leave a half-written artefact that blocks resuming."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import SMOKE_TRAINING

from scaling_lm.config import ResultsPaths, RunConfig
from scaling_lm.runs import CHECKPOINT_FILENAME, RESULT_FILENAME, resolve_run
from scaling_lm.train import load_model, save_run, train_run

HALF = 2


@pytest.fixture
def finished_run(tmp_path, synthetic_corpus):
    """A trained run plus its artefacts, deleted again so the test can redo the save."""
    paths = ResultsPaths(tmp_path)
    run_config = RunConfig("tiny", training=SMOKE_TRAINING)
    result = train_run(run_config, paths)
    model = load_model(run_config.run_name, paths, torch.device("cpu"))
    output_dir = paths.run_directory(run_config.run_name)
    for artefact in output_dir.iterdir():
        artefact.unlink()
    return model, result, run_config, paths


def interrupt_text_writes_to(monkeypatch, filename: str) -> None:
    """Make any text write whose target name contains `filename` stop half way through."""
    original_write_text = Path.write_text

    def interrupted_write_text(self: Path, data: str, *args: object, **kwargs: object) -> int:
        if filename in self.name:
            original_write_text(self, data[: len(data) // HALF], *args, **kwargs)
            raise OSError("connection lost mid-write")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", interrupted_write_text)


def interrupt_checkpoint_writes_to(monkeypatch, filename: str) -> None:
    original_save = torch.save

    def interrupted_save(state: object, path: Path, *args: object, **kwargs: object) -> None:
        if filename in Path(path).name:
            Path(path).write_bytes(b"partial")
            raise OSError("connection lost mid-write")
        original_save(state, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", interrupted_save)


def test_interrupted_result_write_leaves_no_result_file(finished_run, monkeypatch):
    model, result, run_config, paths = finished_run
    output_dir = paths.run_directory(run_config.run_name)
    interrupt_text_writes_to(monkeypatch, RESULT_FILENAME)
    with pytest.raises(OSError, match="mid-write"):
        save_run(model, result, output_dir)
    assert not (output_dir / RESULT_FILENAME).exists()
    assert resolve_run(run_config, paths) is None, "a half-written result blocked resuming"
    assert [path.name for path in output_dir.iterdir()] == [CHECKPOINT_FILENAME]


def test_interrupted_checkpoint_write_leaves_no_checkpoint(finished_run, monkeypatch):
    model, result, run_config, paths = finished_run
    output_dir = paths.run_directory(run_config.run_name)
    interrupt_checkpoint_writes_to(monkeypatch, CHECKPOINT_FILENAME)
    with pytest.raises(OSError, match="mid-write"):
        save_run(model, result, output_dir)
    assert list(output_dir.iterdir()) == []
    assert resolve_run(run_config, paths) is None


def test_interrupted_rewrite_keeps_the_previous_complete_files(finished_run, monkeypatch):
    model, result, run_config, paths = finished_run
    output_dir = paths.run_directory(run_config.run_name)
    save_run(model, result, output_dir)
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    interrupt_text_writes_to(monkeypatch, RESULT_FILENAME)
    with pytest.raises(OSError, match="mid-write"):
        save_run(model, result, output_dir)
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before


def test_completed_save_leaves_only_the_final_files(finished_run):
    model, result, run_config, paths = finished_run
    output_dir = paths.run_directory(run_config.run_name)
    save_run(model, result, output_dir)
    written = sorted(path.name for path in output_dir.iterdir())
    assert written == sorted([CHECKPOINT_FILENAME, RESULT_FILENAME])
    loaded = resolve_run(run_config, paths)
    assert loaded is not None and loaded.final_validation_loss == result.final_validation_loss
    reloaded = load_model(run_config.run_name, paths, torch.device("cpu"))
    assert reloaded.state_dict().keys() == model.state_dict().keys()
