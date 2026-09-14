"""Single-run training loop: shuffled passes over the training split with AdamW and cosine decay.

Run one configuration directly with:

    python -m scaling_lm.train --model-size small [--positional rope] [--seed 1] [--epochs 2]

A run whose result.json already exists is loaded rather than retrained, provided
`runs.resolve_run` accepts it as the same run (same corpus, architecture and schedule).
Artefacts are written to a temporary file and renamed into place, so a save interrupted
by a lost session never leaves a half-written model.pt or result.json behind.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from scaling_lm.budget import steps_per_epoch, token_budget
from scaling_lm.config import (
    MODEL_SIZES_BY_NAME,
    POSITIONAL_SCHEMES,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.dataset import TokenWindows, epoch_batches, sequential_batches
from scaling_lm.model import GPT, GPTConfig
from scaling_lm.runs import (
    CHECKPOINT_FILENAME,
    RESULT_FILENAME,
    EvalPoint,
    RunResult,
    describe_device,
    load_run,
    reject_partial_reuse,
    resolve_run,
    run_identity,
    select_device,
    write_atomically,
)
from scaling_lm.validation import non_negative_int, positive_int, require_positive

logger = logging.getLogger(__name__)


def kaplan_learning_rate(non_embedding_params: int, config: TrainingConfig) -> float:
    """Peak learning rate from Kaplan et al.'s (2020) empirical fit against model size."""
    return config.kaplan_lr_intercept + config.kaplan_lr_slope * math.log(non_embedding_params)


def learning_rate_at(step: int, total_steps: int, peak: float, config: TrainingConfig) -> float:
    """Linear warmup to `peak`, then cosine decay to `config.final_lr_fraction * peak`."""
    warmup_steps = max(1, int(config.warmup_fraction * total_steps))
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    floor = config.final_lr_fraction * peak
    return floor + (peak - floor) * cosine


def build_optimizer(model: GPT, peak_lr: float, config: TrainingConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied to weight matrices only, not biases or norm gains."""
    decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    undecayed = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    groups = [
        {"params": decayed, "weight_decay": config.weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=peak_lr, betas=(config.adam_beta1, config.adam_beta2))


def autocast_context(device: torch.device, enabled: bool) -> torch.autocast:
    use_autocast = enabled and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_autocast)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    windows: TokenWindows,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
    mixed_precision: bool,
) -> float:
    """Mean next-token cross entropy over up to `max_batches` sequential batches of a split."""
    model.eval()
    total_loss = 0.0
    total_windows = 0
    for batch_index, window_indices in enumerate(sequential_batches(windows, batch_size)):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, targets = windows.batch(window_indices, device)
        with autocast_context(device, mixed_precision):
            _, loss = model(inputs, targets)
        total_loss += loss.item() * len(window_indices)
        total_windows += len(window_indices)
    model.train()
    return total_loss / max(1, total_windows)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(run_config: RunConfig, device: torch.device) -> GPT:
    size = MODEL_SIZES_BY_NAME[run_config.model_size]
    return GPT(GPTConfig.from_model_size(size, run_config.positional_scheme)).to(device)


def micro_batches_per_step(config: TrainingConfig) -> int:
    return config.gradient_accumulation_steps


def planned_steps(train_window_count: int, config: TrainingConfig, epochs: int = 1) -> int:
    """Optimiser steps over `epochs` passes (full batches only), capped by max_steps. At least 1."""
    require_positive("epochs", epochs)
    total_steps = steps_per_epoch(train_window_count, config) * epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    if total_steps < 1:
        raise ValueError(
            f"training split has {train_window_count} windows, not enough for one step of "
            f"{config.batch_size_sequences} x {config.gradient_accumulation_steps} sequences"
        )
    return total_steps


def epochs_completed(total_steps: int, epoch_length: int) -> float:
    """Passes actually made, as a fraction of `epoch_length` steps; whole when the plan ran out."""
    require_positive("epoch_length", epoch_length)
    if total_steps < 0:
        raise ValueError(f"total_steps must be non-negative, got {total_steps}")
    return total_steps / epoch_length


def train_step(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    micro_batches: list[tuple[Tensor, Tensor]],
    device: torch.device,
    config: TrainingConfig,
) -> float:
    """One optimiser update over the given micro-batches. Returns the mean training loss."""
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0.0
    for inputs, targets in micro_batches:
        with autocast_context(device, config.use_mixed_precision):
            _, loss = model(inputs, targets)
        (loss / len(micro_batches)).backward()
        accumulated += loss.item() / len(micro_batches)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()
    return accumulated


def train_run(run_config: RunConfig, paths: ResultsPaths) -> RunResult:
    """Train one model for `run_config.epochs` passes over the training split. Saves artefacts."""
    config = run_config.training
    output_dir = paths.run_directory(run_config.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    seed_everything(run_config.seed)

    model = build_model(run_config, device)
    parameter_counts = model.count_parameters()
    peak_lr = kaplan_learning_rate(parameter_counts["non_embedding"], config)
    # torch.compile wraps the module; keep the raw model for parameter counts and saving.
    forward_model: nn.Module = torch.compile(model) if config.compile_model else model

    identity = run_identity(run_config)
    train_windows = TokenWindows("train")
    validation_windows = TokenWindows("validation")
    test_windows = TokenWindows("test")
    total_steps = planned_steps(len(train_windows), config, run_config.epochs)
    logger.info(
        "%s: %s params, %d epochs, %d steps of %d tokens, peak lr %.2e, device %s",
        run_config.run_name,
        parameter_counts,
        run_config.epochs,
        total_steps,
        config.tokens_per_step,
        peak_lr,
        device,
    )

    optimizer = build_optimizer(model, peak_lr, config)
    batch_iterator = epoch_batches(
        train_windows,
        config.batch_size_sequences,
        run_config.seed,
        run_config.epochs,
        micro_batches_per_step(config),
    )
    epoch_length = steps_per_epoch(len(train_windows), config)
    history: list[EvalPoint] = []
    start_time = time.time()
    forward_model.train()
    for step in range(total_steps):
        learning_rate = learning_rate_at(step, total_steps, peak_lr, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        micro_batches = [
            train_windows.batch(next(batch_iterator), device)
            for _ in range(micro_batches_per_step(config))
        ]
        train_loss = train_step(forward_model, optimizer, micro_batches, device, config)

        if step % config.log_interval_steps == 0:
            logger.info(
                "epoch %d/%d step %d/%d loss %.4f lr %.2e",
                step // epoch_length + 1,
                run_config.epochs,
                step,
                total_steps,
                train_loss,
                learning_rate,
            )
        is_last = step == total_steps - 1
        if step % config.eval_interval_steps == 0 or is_last:
            validation_loss = evaluate(
                forward_model,
                validation_windows,
                config.batch_size_sequences,
                device,
                config.eval_batches_periodic,
                config.use_mixed_precision,
            )
            history.append(
                EvalPoint(
                    step=step,
                    tokens_seen=(step + 1) * config.tokens_per_step,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    learning_rate=learning_rate,
                )
            )
            logger.info("step %d validation loss %.4f", step, validation_loss)

    wall_time = time.time() - start_time
    completed = epochs_completed(total_steps, epoch_length)
    if completed < run_config.epochs:
        logger.info(
            "%s: max_steps=%s stopped training after %.3g of %d planned %s (%d steps per pass)",
            run_config.run_name,
            config.max_steps,
            completed,
            run_config.epochs,
            "pass" if run_config.epochs == 1 else "passes",
            epoch_length,
        )
    final_validation = evaluate(
        forward_model,
        validation_windows,
        config.batch_size_sequences,
        device,
        None,
        config.use_mixed_precision,
    )
    final_test = evaluate(
        forward_model,
        test_windows,
        config.batch_size_sequences,
        device,
        None,
        config.use_mixed_precision,
    )
    logger.info(
        "%s finished: validation %.4f test %.4f in %.0fs",
        run_config.run_name,
        final_validation,
        final_test,
        wall_time,
    )

    result = RunResult(
        run_name=run_config.run_name,
        model_size=run_config.model_size,
        positional_scheme=run_config.positional_scheme,
        seed=run_config.seed,
        parameters=parameter_counts,
        identity=identity,
        epochs_completed=completed,
        total_steps=total_steps,
        tokens_seen=total_steps * config.tokens_per_step,
        peak_learning_rate=peak_lr,
        final_validation_loss=final_validation,
        final_test_loss=final_test,
        wall_time_seconds=wall_time,
        device=describe_device(device),
        history=history,
    )
    save_run(model, result, output_dir)
    return result


def save_run(model: GPT, result: RunResult, output_dir: Path) -> None:
    """Persist the checkpoint, then the result. The result is what marks a run finished."""
    write_atomically(
        output_dir / CHECKPOINT_FILENAME, lambda path: torch.save(model.state_dict(), path)
    )
    write_atomically(
        output_dir / RESULT_FILENAME,
        lambda path: path.write_text(json.dumps(asdict(result), indent=2)),
    )


def train_or_load(
    run_config: RunConfig, paths: ResultsPaths, allow_partial: bool = False
) -> RunResult:
    """Return the verified saved result for this run, training it first if there is none.

    A saved result that only made part of its planned passes (`max_steps` cut it short)
    is not handed back for a full request unless `allow_partial` says to; see
    `runs.reject_partial_reuse`.
    """
    existing = resolve_run(run_config, paths)
    if existing is None:
        return train_run(run_config, paths)
    reject_partial_reuse(existing, run_config, paths, allow_partial)
    return existing


def load_model_from_result(result: RunResult, paths: ResultsPaths, device: torch.device) -> GPT:
    """Build and load the checkpoint for an already-verified result. See `load_model`."""
    size = MODEL_SIZES_BY_NAME[result.model_size]
    model = GPT(GPTConfig.from_model_size(size, result.positional_scheme))
    checkpoint_path = paths.run_directory(result.run_name) / CHECKPOINT_FILENAME
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model.to(device).eval()


def load_model(run_name: str, paths: ResultsPaths, device: torch.device) -> GPT:
    """Load a finished run's checkpoint by name, verifying it first through `load_run`."""
    return load_model_from_result(load_run(run_name, paths), paths, device)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = TrainingConfig()
    parser.add_argument(
        "--results-dir", type=Path, default=ResultsPaths().root, help="where to write run artefacts"
    )
    parser.add_argument("--batch-size", type=positive_int, default=defaults.batch_size_sequences)
    parser.add_argument(
        "--grad-accumulation", type=positive_int, default=defaults.gradient_accumulation_steps
    )
    parser.add_argument(
        "--max-steps", type=positive_int, default=None, help="cap steps (smoke tests)"
    )
    parser.add_argument("--eval-interval", type=positive_int, default=defaults.eval_interval_steps)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="reuse a saved result that only made part of its planned passes, for a full "
        "(uncapped) request; without this, such a mismatch is an error",
    )


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        batch_size_sequences=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        max_steps=args.max_steps,
        eval_interval_steps=args.eval_interval,
        use_mixed_precision=not args.no_mixed_precision,
        compile_model=args.compile,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", required=True, choices=sorted(MODEL_SIZES_BY_NAME))
    parser.add_argument("--positional", default="learned", choices=POSITIONAL_SCHEMES)
    parser.add_argument("--seed", type=non_negative_int, default=0)
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=None,
        help="passes over the training split; default: the per-size rule in budget.py",
    )
    add_training_arguments(parser)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    training = training_config_from_args(args)
    epochs = args.epochs
    if epochs is None:
        epochs = token_budget(args.model_size, len(TokenWindows("train")), training).epochs
    run_config = RunConfig(
        model_size=args.model_size,
        positional_scheme=args.positional,
        seed=args.seed,
        training=training,
        epochs=epochs,
    )
    train_or_load(run_config, ResultsPaths(args.results_dir), allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
