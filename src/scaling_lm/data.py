"""Build the arXiv abstract corpus: download, filter to target categories, deduplicate, split.

Run as a script:

    python -m scaling_lm.data [--shards N] [--max-documents N]

Output is one JSONL file per split under data/corpus/ with fields id, categories, text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download

from scaling_lm.config import (
    CORPUS_DIR,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    HF_PARQUET_SHARD_COUNT,
    HF_PARQUET_SHARD_PATTERN,
    MIN_ABSTRACT_CHARS,
    PARQUET_COLUMNS,
    RAW_DIR,
    SPLIT_FRACTIONS,
    SPLIT_HASH_BUCKETS,
    SPLIT_NAMES,
    TARGET_CATEGORIES,
)
from scaling_lm.validation import positive_int

logger = logging.getLogger(__name__)

WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class Document:
    """One arXiv paper reduced to what the language model will see."""

    arxiv_id: str
    categories: str
    text: str


def download_shards(shard_count: int) -> list[Path]:
    """Fetch the first `shard_count` parquet shards at the pinned dataset revision."""
    if not 1 <= shard_count <= HF_PARQUET_SHARD_COUNT:
        raise ValueError(
            f"shard_count must be between 1 and {HF_PARQUET_SHARD_COUNT}, got {shard_count}"
        )
    paths = []
    for index in range(shard_count):
        filename = HF_PARQUET_SHARD_PATTERN.format(index=index)
        logger.info("downloading %s", filename)
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=filename,
            revision=HF_DATASET_REVISION,
            cache_dir=RAW_DIR,
        )
        paths.append(Path(local_path))
    return paths


def normalise_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def has_target_category(categories: str) -> bool:
    return any(category in TARGET_CATEGORIES for category in categories.split())


def build_document_text(title: str, abstract: str) -> str:
    return f"{normalise_whitespace(title)}\n{normalise_whitespace(abstract)}"


def iter_shard_documents(shard_path: Path) -> Iterator[Document]:
    """Yield documents from one shard that fall in the target categories and pass length checks."""
    table = parquet.read_table(shard_path, columns=list(PARQUET_COLUMNS))
    for batch in table.to_batches():
        rows = batch.to_pylist()
        for row in rows:
            categories = row["categories"] or ""
            abstract = row["abstract"] or ""
            if not has_target_category(categories):
                continue
            if len(abstract) < MIN_ABSTRACT_CHARS:
                continue
            yield Document(
                arxiv_id=row["id"],
                categories=categories,
                text=build_document_text(row["title"] or "", abstract),
            )


def stable_hash_bucket(key: str) -> int:
    """Map a string to a bucket in [0, SPLIT_HASH_BUCKETS) using a run-independent hash."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SPLIT_HASH_BUCKETS


def assign_split(arxiv_id: str) -> str:
    """Deterministically place a document in train / validation / test by hashing its id."""
    bucket = stable_hash_bucket(arxiv_id)
    threshold = 0.0
    for split_name in SPLIT_NAMES:
        threshold += SPLIT_FRACTIONS[split_name] * SPLIT_HASH_BUCKETS
        if bucket < threshold:
            return split_name
    return SPLIT_NAMES[-1]


def deduplicate(documents: Iterator[Document]) -> Iterator[Document]:
    """Drop documents whose text has already been seen (exact match after whitespace normalisation).

    arXiv occasionally carries the same abstract under two ids. Removing those
    before splitting is what stops a validation abstract also appearing in training.
    """
    seen_digests: set[bytes] = set()
    for document in documents:
        digest = hashlib.sha1(document.text.encode("utf-8")).digest()
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        yield document


def iter_all_documents(shard_paths: list[Path]) -> Iterator[Document]:
    for shard_path in shard_paths:
        yield from iter_shard_documents(shard_path)
        logger.info("finished scanning %s", shard_path.name)


def collect_documents(shard_paths: list[Path], max_documents: int | None) -> list[Document]:
    documents: list[Document] = []
    for document in deduplicate(iter_all_documents(shard_paths)):
        documents.append(document)
        if max_documents is not None and len(documents) >= max_documents:
            break
    return documents


def split_documents(documents: list[Document]) -> dict[str, list[Document]]:
    """Split by id hash, then order each split by a second hash so the stream is shuffled.

    arXiv ids are chronological. Sorting by hash instead of id means one pass over
    the training stream does not drift from 1990s to 2020s vocabulary.
    """
    splits: dict[str, list[Document]] = {name: [] for name in SPLIT_NAMES}
    for document in documents:
        splits[assign_split(document.arxiv_id)].append(document)
    for split_documents_list in splits.values():
        split_documents_list.sort(key=lambda doc: stable_hash_bucket("order:" + doc.arxiv_id))
    return splits


def write_split(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def read_split(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield Document(**json.loads(line))


def split_path(split_name: str) -> Path:
    return CORPUS_DIR / f"{split_name}.jsonl"


def build_corpus(shard_count: int, max_documents: int | None) -> dict[str, int]:
    """Download, filter, deduplicate, split and write the corpus. Returns document counts."""
    shard_paths = download_shards(shard_count)
    documents = collect_documents(shard_paths, max_documents)
    logger.info("%d unique documents in target categories", len(documents))
    splits = split_documents(documents)
    counts = {}
    for split_name, split_docs in splits.items():
        write_split(split_docs, split_path(split_name))
        counts[split_name] = len(split_docs)
        logger.info("%s: %d documents", split_name, len(split_docs))
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=positive_int, default=HF_PARQUET_SHARD_COUNT)
    parser.add_argument("--max-documents", type=positive_int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    build_corpus(shard_count=args.shards, max_documents=args.max_documents)


if __name__ == "__main__":
    main()
