import hashlib
import json

import numpy as np
import pytest

from scripts.prepare_data import iter_documents, parse_args, prepare, write_split
from src.data import TokenBlockDataset


class TinyTokenizer:
    """Deterministic CPU-only stand-in; never contacts Hugging Face."""

    eos_token_id = 151643

    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


def rows(*texts):
    return [{"text": text} for text in texts]


def test_articles_not_paragraphs_and_sections_define_eos(tmp_path):
    source = rows("", " = First article = \n", "", " Body.\n", " \t\n",
                  " = = Section = = \n", " More.\n", " = Second article = \n", " End.")
    expected = [" = First article = \n Body.\n = = Section = = \n More.\n",
                " = Second article = \n End.\n"]
    assert list(iter_documents(source)) == expected
    path = tmp_path / "train.bin"
    info = write_split(source, TinyTokenizer(), path)
    wanted = [token for doc in expected for token in [*map(ord, doc), 151643]]
    assert np.fromfile(path, dtype="<u4").tolist() == wanted
    assert info["num_documents"] == 2
    assert info["num_tokens"] == len(wanted)
    assert info["num_bytes"] == path.stat().st_size == 4 * len(wanted)
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_repeatable_across_tokenizer_batch_sizes_and_whole_document_limit(tmp_path):
    source = rows(" = A = \n", " first\n", " = B = \n", " second\n",
                  " = C = \n", " third\n")
    paths = [tmp_path / "one.bin", tmp_path / "two.bin"]
    results = [write_split(source, TinyTokenizer(), path, batch_size=batch_size,
                           max_documents=2) for path, batch_size in zip(paths, [1, 8])]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert results[0]["sha256"] == results[1]["sha256"]
    assert results[0]["num_documents"] == 2
    assert np.fromfile(paths[0], dtype="<u4").tolist() == [
        *map(ord, " = A = \n first\n"), 151643,
        *map(ord, " = B = \n second\n"), 151643,
    ]


def test_empty_split_and_missing_eos_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="no nonempty documents"):
        write_split(rows("", " \n"), TinyTokenizer(), tmp_path / "empty.bin")
    tokenizer = TinyTokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="EOS"):
        write_split(rows("text"), tokenizer, tmp_path / "no-eos.bin")


@pytest.fixture
def fake_hub(monkeypatch):
    import datasets
    import transformers

    calls = {}

    def load_tokenizer(model_id, **kwargs):
        calls["model"] = (model_id, kwargs)
        return TinyTokenizer()

    def load_splits(dataset_id, config, **kwargs):
        calls["dataset"] = (dataset_id, config, kwargs)
        return [rows(" = Train = \n", " training text\n"),
                rows(" = Validation = \n", " held-out text\n")]

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(datasets, "load_dataset", load_splits)
    return calls


def test_prepare_to_dataloader_metadata_and_corruption(tmp_path, fake_hub):
    args = parse_args(["--output-dir", str(tmp_path / "data"), "--max-documents", "1"])
    metadata = prepare(args)
    assert json.loads((args.output_dir / "metadata.json").read_text()) == metadata
    assert {path.name for path in args.output_dir.iterdir()} == {
        "train.bin", "validation.bin", "metadata.json"}
    assert metadata["partial"] is True
    assert metadata["model_revision"] == fake_hub["model"][1]["revision"]
    assert metadata["dataset_revision"] == fake_hub["dataset"][2]["revision"]
    assert fake_hub["dataset"][2]["split"] == ["train", "validation"]
    assert metadata["splits"]["train"]["sha256"] != metadata["splits"]["validation"]["sha256"]
    for split in ("train", "validation"):
        dataset = TokenBlockDataset.from_prepared(args.output_dir, split, seq_len=4)
        assert len(dataset) == metadata["splits"][split]["num_tokens"] // 4
        assert dataset[0]["sample_id"] == 0

    path = args.output_dir / "train.bin"
    contents = bytearray(path.read_bytes())
    contents[0] ^= 1  # Same byte count, different hash.
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="SHA256"):
        TokenBlockDataset.from_prepared(args.output_dir, "train", 4)
    path.write_bytes(contents[:-4])
    with pytest.raises(ValueError, match="file size"):
        TokenBlockDataset.from_prepared(args.output_dir, "train", 4)


def test_failure_does_not_publish_partial_splits(tmp_path, fake_hub, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: [rows("valid"), rows("")])
    args = parse_args(["--output-dir", str(tmp_path / "data")])
    with pytest.raises(ValueError, match="no nonempty documents"):
        prepare(args)
    assert list(args.output_dir.iterdir()) == []


def test_existing_data_is_preserved(tmp_path):
    marker = tmp_path / "train.bin"
    marker.write_bytes(b"keep me")
    args = parse_args(["--output-dir", str(tmp_path)])
    with pytest.raises(FileExistsError, match="empty"):
        prepare(args)
    assert marker.read_bytes() == b"keep me"


@pytest.mark.parametrize("arguments", [
    ["--model-revision", "main"], ["--dataset-revision", "short-sha"],
    ["--max-documents", "0"], ["--batch-size", "-1"],
])
def test_cli_rejects_unpinned_versions_and_invalid_sizes(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)
