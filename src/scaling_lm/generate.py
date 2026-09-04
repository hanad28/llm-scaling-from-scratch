"""Sample short continuations from every sweep checkpoint. Qualitative colour only.

python -m scaling_lm.generate [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from scaling_lm.config import (
    GENERATION_MAX_NEW_TOKENS,
    GENERATION_PROMPTS,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_K,
    ResultsPaths,
)
from scaling_lm.model import GPT
from scaling_lm.runs import select_device
from scaling_lm.sweep import read_sweep_manifest
from scaling_lm.tokenizer import load_tokenizer
from scaling_lm.train import load_model
from scaling_lm.validation import require_positive, require_unique

logger = logging.getLogger(__name__)

GENERATION_SEED = 1234


def generate_continuation(
    model: GPT, prompt: str, device: torch.device, max_new_tokens: int = GENERATION_MAX_NEW_TOKENS
) -> str:
    require_positive("max_new_tokens", max_new_tokens)
    require_positive("GENERATION_TEMPERATURE", GENERATION_TEMPERATURE)
    require_positive("GENERATION_TOP_K", GENERATION_TOP_K)
    if not prompt:
        raise ValueError("prompt must not be empty")
    tokenizer = load_tokenizer()
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=GENERATION_TEMPERATURE,
        top_k=GENERATION_TOP_K,
    )
    return tokenizer.decode(output_ids[0].tolist())


def generate_for_runs(run_names: list[str], paths: ResultsPaths) -> dict[str, dict[str, str]]:
    """Return {run_name: {prompt: continuation}} and write it to results/generations.json."""
    require_unique("run_names", run_names)
    device = select_device()
    samples: dict[str, dict[str, str]] = {}
    for run_name in run_names:
        model = load_model(run_name, paths, device)
        torch.manual_seed(GENERATION_SEED)
        samples[run_name] = {
            prompt: generate_continuation(model, prompt, device) for prompt in GENERATION_PROMPTS
        }
        logger.info("sampled %d prompts from %s", len(GENERATION_PROMPTS), run_name)
    paths.generations.write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ResultsPaths().root)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = ResultsPaths(parse_args().results_dir)
    generate_for_runs(read_sweep_manifest(paths), paths)


if __name__ == "__main__":
    main()
