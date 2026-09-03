"""A constant edit anywhere in the training path must stop a saved result being reused.

Monkeypatching one module's copy of a name only proves that the fingerprint reads that
copy, not that the code which trains the model reads the same value. These tests
instead copy the package, edit the constant's source line in place (the size name is
untouched), and run `train_or_load` and `load_run` from that copy in a fresh interpreter.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import scaling_lm
from scaling_lm import runs as runs_module
from scaling_lm import train
from scaling_lm.config import MODEL_SIZES_BY_NAME, RunConfig, TrainingConfig
from scaling_lm.model import GPTConfig

PACKAGE_DIR = Path(scaling_lm.__file__).parent
STUB_CORPUS_FINGERPRINT = "c" * 64
# Modules whose code runs during training. Any tunable they read must be a field of
# GPTConfig or TrainingConfig so that it is part of the fingerprinted specification.
TRAINING_PATH_MODULES = ("dataset.py", "model.py", "positional.py", "train.py")

CHILD_SCRIPT = """
import json
from dataclasses import asdict
from pathlib import Path
import scaling_lm.runs as runs
runs.corpus_fingerprint = lambda: {corpus!r}
from scaling_lm import train
from scaling_lm.config import ResultsPaths, RunConfig, TrainingConfig
run_config = RunConfig("tiny", "rope", 0, TrainingConfig(max_steps=4))
paths = ResultsPaths(Path({results!r}))
if {action!r} == "save":
    specification = runs.run_specification(run_config)
    result = runs.RunResult(
        run_name=run_config.run_name, model_size="tiny", positional_scheme="rope", seed=0,
        parameters={{"total": 1, "embedding": 1, "non_embedding": 1}},
        specification=specification,
        run_fingerprint=runs.fingerprint_specification(specification),
        total_steps=1, tokens_seen=1, peak_learning_rate=1e-3,
        final_validation_loss=3.0, final_test_loss=3.0, wall_time_seconds=1.0, device="cpu",
    )
    run_dir = paths.run_directory(run_config.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / runs.RESULT_FILENAME).write_text(json.dumps(asdict(result)))
    print("SAVED")
else:
    def refuse_to_train(*_args):
        raise AssertionError("train_run was called")
    train.train_run = refuse_to_train
    loaders = {{
        "train_or_load": lambda: train.train_or_load(run_config, paths),
        "load_run": lambda: runs.load_run(run_config.run_name, paths),
    }}
    for label, loader in loaders.items():
        try:
            loader()
        except runs.StaleRunError:
            print(label, "STALE")
        else:
            print(label, "REUSED")
"""
LOADERS = ("train_or_load", "load_run")


def copy_package(destination: Path) -> Path:
    ignore_caches = shutil.ignore_patterns("__pycache__")
    shutil.copytree(PACKAGE_DIR, destination / "scaling_lm", ignore=ignore_caches)
    return destination


def edit_constant(package_root: Path, name: str, new_value: str) -> Path:
    """Rewrite `NAME = <literal>` where the package defines it; fail unless defined exactly once."""
    pattern = re.compile(rf"^{name} = .+$", re.MULTILINE)
    matches = [
        (path, pattern.findall(path.read_text()))
        for path in sorted((package_root / "scaling_lm").glob("*.py"))
    ]
    defining = [(path, found) for path, found in matches if found]
    assert len(defining) == 1 and len(defining[0][1]) == 1, f"{name} must be defined exactly once"
    path, _ = defining[0]
    path.write_text(pattern.sub(f"{name} = {new_value}", path.read_text()))
    return path


def run_child(package_root: Path, results_dir: Path, action: str) -> str:
    script = CHILD_SCRIPT.format(
        corpus=STUB_CORPUS_FINGERPRINT, results=str(results_dir), action=action
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={"PYTHONPATH": str(package_root), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


@pytest.fixture(scope="module")
def saved_by_pristine_copy(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("staleness")
    pristine = copy_package(root / "pristine")
    results_dir = root / "results"
    assert run_child(pristine, results_dir, "save") == "SAVED"
    return pristine, results_dir


def outcomes(child_output: str) -> dict[str, str]:
    return dict(line.split() for line in child_output.splitlines())


def test_unchanged_copy_reuses_the_saved_result(saved_by_pristine_copy):
    pristine, results_dir = saved_by_pristine_copy
    assert outcomes(run_child(pristine, results_dir, "load")) == dict.fromkeys(LOADERS, "REUSED")


@pytest.mark.parametrize(
    ("name", "new_value"),
    [
        ("FREQUENCY_BASE", "500.0"),
        ("INIT_STD", "0.05"),
        ("LAYER_NORM_EPS", "1e-3"),
        ("WARMUP_FRACTION", "0.5"),
        ("KAPLAN_LR_SLOPE", "0.0"),
        ("HEAD_DIM", "32"),
    ],
)
def test_edited_constant_refuses_reuse(saved_by_pristine_copy, tmp_path, name, new_value):
    _, results_dir = saved_by_pristine_copy
    edited = copy_package(tmp_path / "edited")
    edit_constant(edited, name, new_value)
    outcome = outcomes(run_child(edited, results_dir, "load"))
    assert outcome == dict.fromkeys(LOADERS, "STALE"), f"{name} changed but reused: {outcome}"


def module_level_literal_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
    return names


@pytest.mark.parametrize("module", TRAINING_PATH_MODULES)
def test_training_path_modules_keep_no_literal_constants(module):
    constants = module_level_literal_constants(PACKAGE_DIR / module)
    assert constants == [], (
        f"{module} defines {constants}; make them GPTConfig or TrainingConfig fields so "
        "they are part of the fingerprinted run specification"
    )


def test_specification_is_exactly_the_resolved_configs(monkeypatch):
    monkeypatch.setattr(runs_module, "corpus_fingerprint", lambda: STUB_CORPUS_FINGERPRINT)
    run_config = RunConfig("small", "learned", 3, TrainingConfig(max_steps=2))
    specification = runs_module.run_specification(run_config)
    model = GPTConfig.from_model_size(MODEL_SIZES_BY_NAME["small"], "learned")
    assert specification["architecture"] == asdict(model)
    assert specification["schedule"] == {**asdict(run_config.training), "seed": 3}


def test_cli_option_fields_are_exactly_those_the_train_cli_sets():
    parser = argparse.ArgumentParser()
    train.add_training_arguments(parser)
    args = parser.parse_args(
        "--batch-size 7 --grad-accumulation 3 --max-steps 5 --eval-interval 9 "
        "--no-mixed-precision --compile".split()
    )
    from_cli = asdict(train.training_config_from_args(args))
    defaults = asdict(TrainingConfig())
    changed = {name for name, value in from_cli.items() if value != defaults[name]}
    assert changed == TrainingConfig.cli_option_names()
