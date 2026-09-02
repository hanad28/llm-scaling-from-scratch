"""Single-run training loop: one shuffled pass over the training split with AdamW and cosine decay.

Run one configuration directly with:

    python -m scaling_lm.train --model-size small [--positional rope] [--seed 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from scaling_lm.config import (
    CONTEXT_LENGTH,
    FINAL_LR_FRACTION,
    KAPLAN_LR_INTERCEPT,
    KAPLAN_LR_SLOPE,
    MODEL_SIZES_BY_NAME,
    POSITIONAL_SCHEMES,
    WARMUP_FRACTION,
    ResultsPaths,
    RunConfig,
    TrainingConfig,
)
from scaling_lm.dataset import TokenWindows, epoch_batches, sequential_batches
from scaling_lm.model import GPT, GPTConfig

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "model.pt"
RESULT_FILENAME = "result.json"


@dataclass
class EvalPoint:
    step: int
    tokens_seen: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass
class RunResult:
    """Everything recorded about a finished run, written to results/runs/<run_name>/result.json."""

    run_name: str
    model_size: str
    positional_scheme: str
    seed: int
    parameters: dict[str, int]
    architecture: dict[str, int]
    training: dict[str, object]
    total_steps: int
    tokens_seen: int
    peak_learning_rate: float
    final_validation_loss: float
    final_test_loss: float
    wall_time_seconds: float
    device: str
    history: list[EvalPoint] = field(default_factory=list)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kaplan_learning_rate(non_embedding_params: int) -> float:
    """Peak learning rate from Kaplan et al.'s (2020) empirical fit against model size."""
    return KAPLAN_LR_INTERCEPT + KAPLAN_LR_SLOPE * math.log(non_embedding_params)


def learning_rate_at(step: int, total_steps: int, peak: float) -> float:
    """Linear warmup to `peak`, then cosine decay to FINAL_LR_FRACTION * peak."""
    warmup_steps = max(1, int(WARMUP_FRACTION * total_steps))
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    floor = FINAL_LR_FRACTION * peak
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
    """Train one model for one pass over the training split and evaluate it. Saves artefacts."""
    config = run_config.training
    output_dir = paths.run_directory(run_config.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    seed_everything(run_config.seed)

    model = build_model(run_config, device)
    parameter_counts = model.count_parameters()
    peak_lr = kaplan_learning_rate(parameter_counts["non_embedding"])
    # torch.compile wraps the module; keep the raw model for parameter counts and saving.
    forward_model: nn.Module = torch.compile(model) if config.compile_model else model

    train_windows = TokenWindows("train")
    validation_windows = TokenWindows("validation")
    test_windows = TokenWindows("test")
    micro_batches_total = len(train_windows) // config.batch_size_sequences
    total_steps = micro_batches_total // micro_batches_per_step(config)
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    logger.info(
        "%s: %s params, %d steps of %d tokens, peak lr %.2e, device %s",
        run_config.run_name,
        parameter_counts,
        total_steps,
        config.tokens_per_step,
        peak_lr,
        device,
    )

    optimizer = build_optimizer(model, peak_lr, config)
    batch_iterator = epoch_batches(train_windows, config.batch_size_sequences, run_config.seed)
    history: list[EvalPoint] = []
    start_time = time.time()
    forward_model.train()
    for step in range(total_steps):
        learning_rate = learning_rate_at(step, total_steps, peak_lr)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        micro_batches = [
            train_windows.batch(next(batch_iterator), device)
            for _ in range(micro_batches_per_step(config))
        ]
        train_loss = train_step(forward_model, optimizer, micro_batches, device, config)

        if step % config.log_interval_steps == 0:
            logger.info(
                "step %d/%d loss %.4f lr %.2e", step, total_steps, train_loss, learning_rate
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

    size = MODEL_SIZES_BY_NAME[run_config.model_size]
    result = RunResult(
        run_name=run_config.run_name,
        model_size=run_config.model_size,
        positional_scheme=run_config.positional_scheme,
        seed=run_config.seed,
        parameters=parameter_counts,
        architecture={
            "n_layer": size.n_layer,
            "d_model": size.d_model,
            "n_head": size.n_head,
            "context_length": CONTEXT_LENGTH,
        },
        training=asdict(config),
        total_steps=total_steps,
        tokens_seen=total_steps * config.tokens_per_step,
        peak_learning_rate=peak_lr,
        final_validation_loss=final_validation,
        final_test_loss=final_test,
        wall_time_seconds=wall_time,
        device=str(device),
        history=history,
    )
    save_run(model, result, output_dir)
    return result


def save_run(model: GPT, result: RunResult, output_dir: Path) -> None:
    torch.save(model.state_dict(), output_dir / CHECKPOINT_FILENAME)
    (output_dir / RESULT_FILENAME).write_text(json.dumps(asdict(result), indent=2))


def result_exists(run_name: str, paths: ResultsPaths) -> bool:
    return (paths.run_directory(run_name) / RESULT_FILENAME).exists()


def load_result(run_name: str, paths: ResultsPaths) -> RunResult:
    payload = json.loads((paths.run_directory(run_name) / RESULT_FILENAME).read_text())
    payload["history"] = [EvalPoint(**point) for point in payload["history"]]
    return RunResult(**payload)


def load_model(run_name: str, paths: ResultsPaths, device: torch.device) -> GPT:
    result = load_result(run_name, paths)
    size = MODEL_SIZES_BY_NAME[result.model_size]
    model = GPT(GPTConfig.from_model_size(size, result.positional_scheme))
    checkpoint_path = paths.run_directory(run_name) / CHECKPOINT_FILENAME
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = TrainingConfig()
    parser.add_argument(
        "--results-dir", type=Path, default=ResultsPaths().root, help="where to write run artefacts"
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size_sequences)
    parser.add_argument(
        "--grad-accumulation", type=int, default=defaults.gradient_accumulation_steps
    )
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke tests)")
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval_steps)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true")


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
    parser.add_argument("--seed", type=int, default=0)
    add_training_arguments(parser)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run_config = RunConfig(
        model_size=args.model_size,
        positional_scheme=args.positional,
        seed=args.seed,
        training=training_config_from_args(args),
    )
    train_run(run_config, ResultsPaths(args.results_dir))


if __name__ == "__main__":
    main()
