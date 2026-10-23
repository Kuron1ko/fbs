import hashlib
import json
import operator
from pathlib import Path

import numpy as np
import torch


class TokenBlockDataset(torch.utils.data.Dataset):
    """Non-overlapping token blocks; sample_id is the block's file position."""

    def __init__(self, tokens, seq_len):
        super().__init__()
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_blocks = len(self.tokens) // self.seq_len

    @classmethod
    def from_file(cls, path, seq_len):
        """Read little-endian uint32 tokens without loading the whole file."""
        path = Path(path)
        size = path.stat().st_size
        if size % 4:
            raise ValueError(f"{path}: file size is not a multiple of 4 bytes")
        tokens = np.memmap(path, dtype="<u4", mode="r") if size else np.empty(0, dtype="<u4")
        return cls(tokens, seq_len)

    @classmethod
    def from_prepared(cls, data_dir, split, seq_len, *, verify_checksum=True):
        """Load a split and validate its manifest, size, and (by default) SHA256."""
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        data_dir = Path(data_dir)
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("schema_version") != 1 or metadata.get("dtype") != "uint32"
                or metadata.get("byte_order") != "little"):
            raise ValueError("Unsupported token data format")
        info = metadata["splits"][split]
        if info["file"] != f"{split}.bin":
            raise ValueError("Unexpected split filename in metadata")
        path = data_dir / info["file"]
        size = path.stat().st_size
        if size != info["num_bytes"] or size != info["num_tokens"] * 4:
            raise ValueError(f"{split}: file size does not match metadata")
        if verify_checksum:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != info["sha256"]:
                raise ValueError(f"{split}: SHA256 mismatch")
        return cls.from_file(path, seq_len)

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0 or index >= self.num_blocks:
            raise IndexError("block index out of range")
        start = index * self.seq_len
        block = self.tokens[start:start + self.seq_len]
        # Copy the read-only mmap so callers cannot mutate the token file.
        input_ids = torch.tensor(block, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "sample_id": index,
        }

    def __len__(self):
        return self.num_blocks
