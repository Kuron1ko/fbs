import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.data import TokenBlockDataset


def test_blocks_and_types():
    tokens = np.arange(10, dtype=np.uint32)
    dataset = TokenBlockDataset(tokens, seq_len=4)

    assert len(dataset) == 2  # 尾部两个 token 丢弃
    first = dataset[0]
    assert set(first) == {"input_ids", "labels", "sample_id"}
    assert first["sample_id"] == 0
    assert dataset[1]["sample_id"] == 1
    assert first["input_ids"].dtype == torch.long
    assert first["input_ids"].shape == (4,)
    assert torch.equal(first["input_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(first["labels"], first["input_ids"])
    assert torch.equal(dataset[1]["input_ids"], torch.tensor([4, 5, 6, 7]))


def test_invalid_inputs():
    tokens = np.arange(8, dtype=np.uint32)
    with pytest.raises(ValueError):
        TokenBlockDataset(tokens, seq_len=0)

    dataset = TokenBlockDataset(tokens, seq_len=4)
    with pytest.raises(IndexError):
        dataset[2]


def test_two_ranks_do_not_repeat_samples():
    dataset = TokenBlockDataset(np.arange(40, dtype=np.uint32), seq_len=4)
    rank0 = set(DistributedSampler(dataset, 2, 0, shuffle=False, drop_last=True))
    rank1 = set(DistributedSampler(dataset, 2, 1, shuffle=False, drop_last=True))

    assert rank0.isdisjoint(rank1)
    assert rank0 | rank1 == set(range(len(dataset)))


@pytest.mark.parametrize("seq_len", [0, -1, 1.5, True])
def test_rejects_invalid_block_length(seq_len):
    with pytest.raises(ValueError, match="positive integer"):
        TokenBlockDataset(np.arange(8), seq_len)


def test_binary_round_trip_and_independent_tensors(tmp_path):
    path = tmp_path / "train.bin"
    tokens = np.array([0, 1, 151643, 70000, 4, 5, 6, 7, 8], dtype="<u4")
    tokens.tofile(path)
    dataset = TokenBlockDataset.from_file(path, seq_len=4)
    assert isinstance(dataset.tokens, np.memmap)
    assert len(dataset) == 2
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["input_ids"].shape == (2, 4)
    assert batch["sample_id"].tolist() == [0, 1]
    assert batch["input_ids"].flatten().tolist() == tokens[:8].tolist()
    first = dataset[0]
    first["input_ids"][0] = 999
    assert first["labels"][0].item() == 0
    assert dataset[0]["input_ids"][0].item() == 0
    with pytest.raises(IndexError):
        dataset[-1]
    with pytest.raises(TypeError):
        dataset[0.5]


def test_empty_and_truncated_binary_files(tmp_path):
    path = tmp_path / "empty.bin"
    path.touch()
    dataset = TokenBlockDataset.from_file(path, 4)
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        dataset[0]
    path.write_bytes(b"123")
    with pytest.raises(ValueError, match="multiple of 4"):
        TokenBlockDataset.from_file(path, 4)


def test_odd_rank_partition_and_resumed_sample_ids():
    dataset = TokenBlockDataset(np.arange(44), seq_len=4)  # 11 blocks

    def batches(rank, epoch):
        sampler = DistributedSampler(dataset, 2, rank, seed=42, drop_last=True)
        sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
        return [batch["sample_id"].item() for batch in loader]

    rank0, rank1 = batches(0, 3), batches(1, 3)
    assert len(rank0) == len(rank1) == 5
    assert set(rank0).isdisjoint(rank1)
    assert batches(0, 3)[2:] == rank0[2:]  # Rebuild epoch, skip consumed batches.
    assert batches(0, 4) != rank0
