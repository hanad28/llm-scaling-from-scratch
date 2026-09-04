"""Project-wide constants: paths, dataset source, model sizes and training hyperparameters.

Everything tunable lives here so that the experiment scripts contain no magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from scaling_lm.validation import (
    require_fraction,
    require_non_negative,
    require_positive,
    require_unit_interval,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORPUS_DIR = DATA_DIR / "corpus"
TOKENS_DIR = DATA_DIR / "tokens"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
CORPUS_STATS_PATH = DATA_DIR / "corpus_stats.json"
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class ResultsPaths:
    """Layout of one results directory (the main sweep and the CPU pilot use separate roots)."""

    root: Path = RESULTS_DIR

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def sweep_summary(self) -> Path:
        return self.root / "scaling_sweep.json"

    @property
    def ablation_summary(self) -> Path:
        return self.root / "positional_ablation.json"

    @property
    def scaling_fit(self) -> Path:
        return self.root / "scaling_fit.json"

    @property
    def generations(self) -> Path:
        return self.root / "generations.json"

    @property
    def summary_markdown(self) -> Path:
        return self.root / "summary.md"

    def run_directory(self, run_name: str) -> Path:
        return self.runs / run_name


# ---------------------------------------------------------------------------
# Dataset source
# ---------------------------------------------------------------------------
# Weekly mirror of the Cornell arXiv metadata snapshot. The revision is pinned
# so the corpus can be rebuilt byte-for-byte later.
HF_DATASET_REPO = "librarian-bots/arxiv-metadata-snapshot"
HF_DATASET_REVISION = "47141d6fd17f52b65424d246665334914cac3011"
HF_PARQUET_SHARD_PATTERN = "data/train-{index:05d}-of-00010.parquet"
HF_PARQUET_SHARD_COUNT = 10
PARQUET_COLUMNS = ("id", "title", "categories", "abstract")

TARGET_CATEGORIES = frozenset({"cs.AI", "cs.LG", "cs.CL"})

# Abstracts shorter than this (in characters) are usually withdrawal notices or
# placeholders rather than real abstracts.
MIN_ABSTRACT_CHARS = 200

# Fractions of documents per split. Assignment is by hashing the arXiv id, so
# it is deterministic and independent of document order.
SPLIT_FRACTIONS = {"train": 0.98, "validation": 0.01, "test": 0.01}
SPLIT_HASH_BUCKETS = 10_000

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
VOCAB_SIZE = 8192
EOS_TOKEN = "<|endoftext|>"
# Byte-level BPE with 8192 merges comfortably fits in uint16 storage.
TOKEN_DTYPE = "uint16"
TOKENIZER_TRAINING_DOCS = 200_000

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Every constant the model code reads is a GPTConfig field defaulting to one of these,
# so that the resolved GPTConfig is a complete record of the architecture.
CONTEXT_LENGTH = 256
HEAD_DIM = 64
MLP_EXPANSION = 4
POSITIONAL_SCHEMES = ("learned", "sinusoidal", "rope")
DEFAULT_POSITIONAL_SCHEME = "learned"
# GPT-2 initialisation scale (Radford et al., 2019).
INIT_STD = 0.02
LAYER_NORM_EPS = 1e-5
# Base of the geometric frequency progression used by Vaswani et al. (2017) and
# retained by Su et al. (2024) for RoPE.
FREQUENCY_BASE = 10_000.0


@dataclass(frozen=True)
class ModelSize:
    """Architecture hyperparameters for one point on the size sweep."""

    name: str
    n_layer: int
    d_model: int

    @property
    def n_head(self) -> int:
        return self.d_model // HEAD_DIM

    @property
    def approx_non_embedding_params(self) -> int:
        """Kaplan et al.'s 12 * n_layer * d_model^2 estimate (attention + MLP weights)."""
        return 12 * self.n_layer * self.d_model**2


# Head dimension is fixed at 64 and the MLP ratio at 4, so width and depth are
# the only things that change. d_model / n_layer stays in the 32 to 64 band.
# Roughly log-spaced from 0.8M to 99M non-embedding parameters, in ascending order.
MODEL_SIZES: tuple[ModelSize, ...] = (
    ModelSize(name="tiny", n_layer=4, d_model=128),
    ModelSize(name="small", n_layer=6, d_model=256),
    ModelSize(name="medium", n_layer=7, d_model=384),
    ModelSize(name="large", n_layer=8, d_model=512),
    ModelSize(name="xlarge", n_layer=10, d_model=640),
    ModelSize(name="xxlarge", n_layer=14, d_model=768),
)
MODEL_SIZES_BY_NAME = {size.name: size for size in MODEL_SIZES}

# The ablation runs at a single mid-sized point on the sweep.
ABLATION_MODEL_SIZE = "medium"
ABLATION_POSITIONAL_SCHEMES = ("learned", "rope")
ABLATION_SEEDS = (0, 1, 2)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# As with the model, every constant the training loop reads is a TrainingConfig field.
# Kaplan et al. (2020, appendix D.6) fitted lr(N) = 0.003239 - 0.0001395 ln(N)
# for non-embedding parameter count N. The rule is used as-is for every size.
KAPLAN_LR_INTERCEPT = 0.003239
KAPLAN_LR_SLOPE = -0.0001395

# Cosine decay finishes at this fraction of the peak learning rate.
FINAL_LR_FRACTION = 0.1
WARMUP_FRACTION = 0.05


# Marks the TrainingConfig fields a command line may set per invocation. When a saved
# run is re-checked by name, only these are taken from the saved record; every other
# field comes from the current code, so a changed constant is detected on that path too.
CLI_OPTION = {"cli_option": True}


@dataclass(frozen=True)
class TrainingConfig:
    """Optimiser and schedule settings shared by every run in the sweep."""

    batch_size_sequences: int = field(default=128, metadata=CLI_OPTION)
    gradient_accumulation_steps: int = field(default=1, metadata=CLI_OPTION)
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip_norm: float = 1.0
    kaplan_lr_intercept: float = KAPLAN_LR_INTERCEPT
    kaplan_lr_slope: float = KAPLAN_LR_SLOPE
    warmup_fraction: float = WARMUP_FRACTION
    final_lr_fraction: float = FINAL_LR_FRACTION
    eval_interval_steps: int = field(default=200, metadata=CLI_OPTION)
    # Batches of validation data used for the periodic (cheap) evaluation.
    eval_batches_periodic: int = 20
    use_mixed_precision: bool = field(default=True, metadata=CLI_OPTION)
    compile_model: bool = field(default=False, metadata=CLI_OPTION)
    log_interval_steps: int = 50
    # Set to cap the number of training steps (used for smoke tests only).
    max_steps: int | None = field(default=None, metadata=CLI_OPTION)

    def __post_init__(self) -> None:
        require_positive("batch_size_sequences", self.batch_size_sequences)
        require_positive("gradient_accumulation_steps", self.gradient_accumulation_steps)
        require_positive("eval_interval_steps", self.eval_interval_steps)
        require_positive("eval_batches_periodic", self.eval_batches_periodic)
        require_positive("log_interval_steps", self.log_interval_steps)
        require_positive("grad_clip_norm", self.grad_clip_norm)
        require_non_negative("weight_decay", self.weight_decay)
        require_unit_interval("adam_beta1", self.adam_beta1)
        require_unit_interval("adam_beta2", self.adam_beta2)
        require_unit_interval("warmup_fraction", self.warmup_fraction)
        require_fraction("final_lr_fraction", self.final_lr_fraction)
        if self.max_steps is not None:
            require_positive("max_steps", self.max_steps)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size_sequences * self.gradient_accumulation_steps * CONTEXT_LENGTH

    @classmethod
    def cli_option_names(cls) -> frozenset[str]:
        return frozenset(entry.name for entry in fields(cls) if entry.metadata.get("cli_option"))


DEFAULT_TRAINING_CONFIG = TrainingConfig()

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
# Kaplan et al. (2020) report L(N) = (N_c / N)^alpha_N with alpha_N = 0.076.
KAPLAN_ALPHA_N = 0.076
KAPLAN_N_C = 8.8e13
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
GENERATION_MAX_NEW_TOKENS = 120
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 40
GENERATION_PROMPTS: tuple[str, ...] = (
    "We propose a novel",
    "Large language models",
    "In this paper, we study the problem of",
)


@dataclass
class RunConfig:
    """Everything needed to reproduce a single training run.

    `epochs` is the number of shuffled passes over the training split. It sits here rather
    than in TrainingConfig because it is set per size by the sweep (see budget.py), whereas
    TrainingConfig is shared by every run.
    """

    model_size: str
    positional_scheme: str = DEFAULT_POSITIONAL_SCHEME
    seed: int = 0
    training: TrainingConfig = field(default_factory=TrainingConfig)
    epochs: int = 1

    def __post_init__(self) -> None:
        if self.model_size not in MODEL_SIZES_BY_NAME:
            raise ValueError(
                f"unknown model size {self.model_size!r}, expected one of "
                f"{sorted(MODEL_SIZES_BY_NAME)}"
            )
        if self.positional_scheme not in POSITIONAL_SCHEMES:
            raise ValueError(
                f"unknown positional scheme {self.positional_scheme!r}, expected one of "
                f"{list(POSITIONAL_SCHEMES)}"
            )
        require_non_negative("seed", self.seed)
        require_positive("epochs", self.epochs)

    @property
    def run_name(self) -> str:
        return f"{self.model_size}_{self.positional_scheme}_seed{self.seed}"
