"""Train a byte-level BPE tokenizer on the training split and encode every split to a token file.

Run as a script after `python -m scaling_lm.data`:

    python -m scaling_lm.tokenizer

Each split becomes a flat uint16 array of token ids under data/tokens/, with an
end-of-text token between documents. Token counts are written to data/corpus_stats.json.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scaling_lm.config import (
    CORPUS_STATS_PATH,
    EOS_TOKEN,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    SPLIT_NAMES,
    TOKEN_DTYPE,
    TOKENIZER_PATH,
    TOKENIZER_TRAINING_DOCS,
    TOKENS_DIR,
    VOCAB_SIZE,
)
from scaling_lm.data import Document, read_split, split_path

logger = logging.getLogger(__name__)

ENCODE_BATCH_SIZE = 2_000
FINGERPRINT_CHUNK_BYTES = 1 << 24


def train_tokenizer(texts: Iterable[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Train a GPT-2 style byte-level BPE tokenizer with a single special end-of-text token."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def eos_token_id(tokenizer: Tokenizer) -> int:
    token_id = tokenizer.token_to_id(EOS_TOKEN)
    if token_id is None:
        raise ValueError(f"tokenizer has no {EOS_TOKEN} token")
    return token_id


def batched(iterable: Iterable[Document], size: int) -> Iterator[list[Document]]:
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def tokens_path(split_name: str) -> Path:
    return TOKENS_DIR / f"{split_name}.bin"


def encode_split(tokenizer: Tokenizer, documents: Iterable[Document], output_path: Path) -> int:
    """Encode documents into a flat token file, appending EOS after each. Returns token count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eos = np.array([eos_token_id(tokenizer)], dtype=TOKEN_DTYPE)
    total_tokens = 0
    with output_path.open("wb") as handle:
        for batch in batched(documents, ENCODE_BATCH_SIZE):
            encodings = tokenizer.encode_batch([document.text for document in batch])
            for encoding in encodings:
                ids = np.asarray(encoding.ids, dtype=TOKEN_DTYPE)
                handle.write(ids.tobytes())
                handle.write(eos.tobytes())
                total_tokens += len(ids) + 1
    return total_tokens


def load_tokens(split_name: str) -> np.memmap:
    return np.memmap(tokens_path(split_name), dtype=TOKEN_DTYPE, mode="r")


def corpus_fingerprint() -> str:
    """SHA-256 over corpus_stats.json and every split's token file, identifying the exact corpus."""
    digest = hashlib.sha256(CORPUS_STATS_PATH.read_bytes())
    for split_name in SPLIT_NAMES:
        with tokens_path(split_name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(FINGERPRINT_CHUNK_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


def training_texts(limit: int) -> Iterator[str]:
    for document in itertools.islice(read_split(split_path("train")), limit):
        yield document.text


def build_tokens() -> dict[str, object]:
    """Train the tokenizer on the training split only, then encode all splits."""
    logger.info("training tokenizer on up to %d training documents", TOKENIZER_TRAINING_DOCS)
    tokenizer = train_tokenizer(training_texts(TOKENIZER_TRAINING_DOCS))
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))

    stats: dict[str, object] = {
        "dataset_repo": HF_DATASET_REPO,
        "dataset_revision": HF_DATASET_REVISION,
        "vocab_size": tokenizer.get_vocab_size(),
        "documents": {},
        "tokens": {},
    }
    for split_name in SPLIT_NAMES:
        documents = list(read_split(split_path(split_name)))
        token_count = encode_split(tokenizer, documents, tokens_path(split_name))
        stats["documents"][split_name] = len(documents)
        stats["tokens"][split_name] = token_count
        logger.info("%s: %d documents, %d tokens", split_name, len(documents), token_count)

    CORPUS_STATS_PATH.write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_tokens()


if __name__ == "__main__":
    main()
