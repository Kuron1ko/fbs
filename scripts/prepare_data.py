"""Prepare pinned WikiText raw splits for causal language-model training.

Run from the repository root: python scripts/prepare_data.py --help
Only tokenizer files are downloaded from the model repository, never weights.
"""

import argparse
import hashlib
import json
import re
import tempfile
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import numpy as np


MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
ARTICLE_TITLE = re.compile(r"=\s+[^=\s].*?\s+=")
TEXT_POLICY = {
    "document_boundary": "WikiText level-1 title (= Title =); keep section headings inside article",
    "empty_rows": "discard whitespace-only rows",
    "text": "preserve nonempty rows except trailing CR/LF; join with LF and end article with LF",
    "special_tokens": "add_special_tokens=False; append one eos_token_id per article, including last",
    "order": "original split order; no shuffling or mixing train/validation",
}


def iter_documents(rows):
    """Reconstruct articles from WikiText rows, including titles and sections."""
    lines = []
    for row in rows:
        text = row["text"]
        if not isinstance(text, str):
            raise ValueError("Every WikiText row must have a string 'text' field")
        if not text.strip():
            continue
        if ARTICLE_TITLE.fullmatch(text.strip()) and lines:
            yield "\n".join(lines) + "\n"
            lines = []
        lines.append(text.rstrip("\r\n"))
    if lines:
        yield "\n".join(lines) + "\n"


def write_split(rows, tokenizer, path, *, batch_size=8, max_documents=None):
    """Encode bounded batches of whole articles, returning file statistics."""
    if batch_size <= 0 or (max_documents is not None and max_documents <= 0):
        raise ValueError("batch_size and max_documents must be positive")
    eos_id = tokenizer.eos_token_id
    if eos_id is None or not 0 <= eos_id <= np.iinfo(np.uint32).max:
        raise ValueError("Tokenizer must have an EOS token representable as uint32")
    documents = iter_documents(rows)
    if max_documents is not None:
        documents = islice(documents, max_documents)
    digest = hashlib.sha256()
    num_tokens = num_documents = 0
    path = Path(path)
    with path.open("xb") as stream:
        while batch := list(islice(documents, batch_size)):
            encoded = tokenizer(
                batch, add_special_tokens=False, padding=False, truncation=False,
                return_attention_mask=False, return_token_type_ids=False,
            )["input_ids"]
            if len(encoded) != len(batch):
                raise ValueError("Tokenizer returned an unexpected batch size")
            for ids in encoded:
                if ids and (min(ids) < 0 or max(ids) > np.iinfo(np.uint32).max):
                    raise ValueError("Token ID is outside the uint32 range")
                payload = np.asarray([*ids, eos_id], dtype="<u4").tobytes()
                stream.write(payload)
                digest.update(payload)
                num_tokens += len(ids) + 1
                num_documents += 1
            if num_documents % (batch_size * 100) == 0:
                print(f"{path.stem}: {num_documents:,} documents, {num_tokens:,} tokens", flush=True)
    if not num_documents:
        raise ValueError(f"{path.stem}: no nonempty documents found")
    return {
        "file": path.name,
        "num_documents": num_documents,
        "num_tokens": num_tokens,
        "num_bytes": num_tokens * 4,
        "sha256": digest.hexdigest(),
    }


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def commit_sha(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("must be a full 40-character commit SHA")
    return value.lower()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", type=commit_sha, default=MODEL_REVISION)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1",
                        choices=["wikitext-103-raw-v1", "wikitext-2-raw-v1"])
    parser.add_argument("--dataset-revision", type=commit_sha, default=DATASET_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wikitext-103"))
    parser.add_argument("--batch-size", type=positive_int, default=8,
                        help="number of complete articles per tokenizer batch")
    parser.add_argument("--max-documents", type=positive_int,
                        help="tokenize at most this many articles per split (smoke only)")
    return parser.parse_args(argv)


def prepare(args):
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    # Lazy imports keep --help and synthetic tests independent of Hub downloads.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Tokenizer: {args.model_id}@{args.model_revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    print(f"Dataset: {args.dataset_id}@{args.dataset_revision}", flush=True)
    # Disk-backed Arrow avoids keeping the corpus in RAM. Full source splits are
    # cached even when --max-documents limits the amount subsequently tokenized.
    raw_splits = load_dataset(
        args.dataset_id, args.dataset_config, revision=args.dataset_revision,
        split=["train", "validation"],
    )
    metadata = {
        "schema_version": 1,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dtype": "uint32",
        "byte_order": "little",
        "eos_token_id": tokenizer.eos_token_id,
        "tokenizer_class": type(tokenizer).__name__,
        "text_policy": TEXT_POLICY,
        "partial": args.max_documents is not None,
        "max_documents_per_split": args.max_documents,
        "tokenizer_batch_size": args.batch_size,
        "versions": {name: version(name) for name in (
            "transformers", "tokenizers", "datasets", "huggingface_hub", "numpy")},
        "prepare_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": {},
    }
    # Publish metadata last: its presence indicates both splits completed.
    # Encoding/download failures leave no apparently complete output files.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir) as staging:
        staging = Path(staging)
        for split, rows in zip(("train", "validation"), raw_splits):
            info = write_split(
                rows, tokenizer, staging / f"{split}.bin",
                batch_size=args.batch_size, max_documents=args.max_documents,
            )
            metadata["splits"][split] = info
            print(f"{split}: {info['num_documents']:,} documents, "
                  f"{info['num_tokens']:,} tokens; SHA256={info['sha256']}", flush=True)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ("train.bin", "validation.bin", "metadata.json"):
            (staging / name).replace(output_dir / name)
    print(f"Done: {output_dir / 'metadata.json'}", flush=True)
    return metadata


def main():
    args = parse_args()
    try:
        prepare(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Data preparation failed: {exc}") from exc


if __name__ == "__main__":
    main()
