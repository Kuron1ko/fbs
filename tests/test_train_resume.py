"""用 CPU 假 engine 执行真实训练入口，检查样本、随机数序列和停止位置。"""

import hashlib
import json
import logging
from pathlib import Path
import random
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.config import load_config


def write_toml(path, config):
    """写入实际 TOML，测试不绕过配置文件读取。"""
    def value(item):
        if isinstance(item, dict):
            return "{ " + ", ".join(f"{json.dumps(key)} = {value(val)}" for key, val in item.items()) + " }"
        return json.dumps(item)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{key} = {value(item)}" for key, item in config.items()) + "\n")


def load_test_state(path):
    # 文件由本测试生成；完整 RNG 状态包含 NumPy 对象，不能只加载权重。
    return torch.load(path, map_location="cpu", weights_only=False)


class FakeEngine:
    def __init__(self, accumulation, lr):
        self.device = torch.device("cpu")
        self.global_steps = 0
        self.micro_batches = 0
        self.accumulation = accumulation
        self.samples = []
        self.random_draws = []
        self.lr = lr
        self.checkpoint_loads = 0
        self.batch_sizes = []

    def train(self):
        pass

    def get_lr(self):
        return [self.lr]

    def __call__(self, input_ids, **kwargs):
        self.batch_sizes.append(input_ids.shape[0])
        self.samples.append(input_ids[0, 0].item() // 128)
        # 模拟训练中的随机运算，让 RNG 恢复错误能被测试发现。
        self.random_draws.append((random.random(), float(np.random.random()), torch.rand(()).item()))
        return SimpleNamespace(loss=torch.tensor(1.0))

    def backward(self, loss):
        pass

    def step(self):
        self.micro_batches += 1
        if self.micro_batches % self.accumulation == 0:
            self.global_steps += 1

    def save_checkpoint(self, save_dir, tag, client_state):
        assert self.micro_batches % self.accumulation == 0
        assert "global_steps" not in client_state
        directory = Path(save_dir) / tag
        directory.mkdir(parents=True, exist_ok=True)
        state = {"global_steps": self.global_steps, "client_state": client_state}
        torch.save(state, directory / "state.pt")
        (Path(save_dir) / "latest").write_text(tag)

    def load_checkpoint(self, load_dir, tag):
        self.checkpoint_loads += 1
        path = Path(load_dir) / tag / "state.pt"
        state = load_test_state(path)
        self.global_steps = state["global_steps"]
        return str(path), state["client_state"]


@pytest.fixture
def training_entry(monkeypatch, tmp_path):
    engines = []
    evaluations = []
    world_size = [1]
    model_config = {"hidden_size": 128, "num_hidden_layers": 2}
    project_dir = Path(__file__).resolve().parents[1]
    # 即使测试主机有 GPU，这些假 engine 测试也只使用 CPU。
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size[0])
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def all_gather_object(states, state):
        states[:] = [state] * world_size[0]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    def initialize(**kwargs):
        config = kwargs["config"]
        accumulation = config.get("gradient_accumulation_steps")
        if accumulation is None:
            accumulation = config["train_batch_size"] // (config["train_micro_batch_size_per_gpu"] * world_size[0])
        engine = FakeEngine(accumulation, config["optimizer"]["params"]["lr"])
        engine.config = config
        engines.append(engine)
        return engine, None, None, None

    def evaluate(engine, loader):
        evaluations.append(engine.global_steps)
        # 与真实评估一样创建验证集迭代器，覆盖其对 CPU RNG 的影响。
        for _ in loader:
            pass
        return 1.0, 2.718, 1016

    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(initialize=initialize))
    monkeypatch.setitem(sys.modules, "src.model", SimpleNamespace(
        build_tiny_model=lambda **kwargs: SimpleNamespace(
            parameters=lambda: [], config=SimpleNamespace(to_dict=lambda: dict(model_config)))))
    monkeypatch.setitem(sys.modules, "src.evaluate", SimpleNamespace(evaluate=evaluate))
    namespace = runpy.run_path(str(project_dir / "train.py"))
    main = namespace["main"]
    monkeypatch.setitem(main.__globals__, "getlog", lambda: logging.getLogger("test.train"))

    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "smoke"
    data_dir.mkdir(parents=True)

    def run(directory, *, samples=64, max_updates=20, resume_tag=None, seed=42,
            ds_config=None, num_ranks=1, token_offset=0, seq_len=128):
        world_size[0] = num_ranks
        monkeypatch.setenv("WORLD_SIZE", str(num_ranks))
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(sys, "argv", ["train.py", "--local_rank=0"])
        config = load_config(project_dir / "configs/train.toml")
        config["profiler"]["enabled"] = False  # CPU 恢复测试不采集 GPU trace。
        config["data"].update(dir=str(data_dir), seq_len=seq_len)
        config["train"].update(seed=seed, max_updates=max_updates, evaluate_interval=5, save_interval=10)
        config["checkpoint"] = {"dir": str(directory), "resume_tag": resume_tag or ""}
        # 使用实际的小型 token 文件和 SHA256，覆盖恢复检查中的数据身份校验。
        metadata = {
            "schema_version": 1, "dtype": "uint32", "byte_order": "little", "splits": {},
            "model_id": config["model"]["id"], "model_revision": config["model"]["revision"],
            "dataset_id": config["data"]["id"], "dataset_config": config["data"]["config"],
            "dataset_revision": config["data"]["revision"],
        }
        for split, count in (("train", samples), ("validation", 8)):
            tokens = np.arange(count * 128, dtype="<u4")
            if split == "train":
                tokens += token_offset
            path = data_dir / f"{split}.bin"
            tokens.tofile(path)
            metadata["splits"][split] = {
                "file": path.name, "num_tokens": len(tokens), "num_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (data_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        if ds_config is None:
            ds_config = json.loads((project_dir / "configs/deepspeed.json").read_text())
            ds_config.pop("train_batch_size", None)
            ds_config["train_micro_batch_size_per_gpu"] = 1  # 固定样本位置断言的测试条件。
            ds_config["gradient_accumulation_steps"] = 2
            ds_config["scheduler"]["params"]["total_num_steps"] = 20
        write_toml(tmp_path / "configs/train.toml", config)
        (tmp_path / "configs/deepspeed.json").write_text(json.dumps(ds_config))
        main()
        return engines[-1]

    return SimpleNamespace(run=run, evaluations=evaluations, engines=engines, model_config=model_config)


@pytest.mark.parametrize("samples", [64, 6, 5])
def test_resumed_samples_match_uninterrupted_run(training_entry, tmp_path, samples):
    # 分别覆盖 epoch 中间、已经跨过 epoch，以及刚好耗尽 epoch 时的恢复。
    full = training_entry.run(tmp_path / "full", samples=samples)
    first = training_entry.run(tmp_path / "split", samples=samples, max_updates=10)
    path = tmp_path / "split" / "step10" / "state.pt"
    state = load_test_state(path)["client_state"]
    assert state["epoch"] == 19 // samples
    assert state["batches_in_epoch"] == 19 % samples + 1
    assert isinstance(state["rng_state"][0]["numpy"][1], np.ndarray)
    assert isinstance(state["rng_state"][0]["torch_cpu"], torch.Tensor)
    for tag in ("step0", "step10"):
        assert (tmp_path / "split" / tag / "resume_config.json").is_file()

    # 即使 latest 指向不存在的目录，明确指定 step10 也应能加载。
    (tmp_path / "split" / "latest").write_text("step20")
    training_entry.evaluations.clear()
    resumed = training_entry.run(tmp_path / "split", samples=samples, resume_tag="step10")
    assert first.samples + resumed.samples == full.samples
    assert first.random_draws + resumed.random_draws == full.random_draws
    assert resumed.samples[0] == full.samples[20]
    assert resumed.global_steps == 20
    assert training_entry.evaluations == [10, 15, 20]
    final = load_test_state(tmp_path / "split" / "step20" / "state.pt")["client_state"]
    assert final["epoch"] == 39 // samples
    assert final["batches_in_epoch"] == 39 % samples + 1
    assert (tmp_path / "split" / "step20" / "resume_config.json").is_file()


def test_completed_target_does_not_consume_a_batch(training_entry, tmp_path):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    training_entry.evaluations.clear()
    resumed = training_entry.run(directory, max_updates=10, resume_tag="step10")
    assert resumed.samples == []
    assert resumed.random_draws == []
    assert resumed.global_steps == 10
    assert training_entry.evaluations == [10]


def test_step_zero_restores_same_samples_and_random_sequence(training_entry, tmp_path):
    directory = tmp_path / "checkpoints"
    first = training_entry.run(directory, max_updates=10)
    resumed = training_entry.run(directory, max_updates=10, resume_tag="step0")
    assert resumed.samples == first.samples
    assert resumed.random_draws == first.random_draws


@pytest.mark.parametrize("field", ["batches_in_epoch", "rng_state"])
def test_required_resume_state_cannot_be_silently_omitted(training_entry, tmp_path, field):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    path = directory / "step10" / "state.pt"
    state = load_test_state(path)
    del state["client_state"][field]
    torch.save(state, path)
    # 当前入口直接读取必需字段；缺字段时由字典访问抛出 KeyError。
    with pytest.raises(KeyError, match=field):
        training_entry.run(directory, resume_tag="step10")


def test_new_run_preserves_existing_checkpoints_and_missing_tag_is_rejected(training_entry, tmp_path):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    path = directory / "step10" / "state.pt"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="从头训练"):
        training_entry.run(directory)
    with pytest.raises(ValueError, match="找不到 checkpoint"):
        training_entry.run(directory, resume_tag="step20")
    assert path.read_bytes() == before


@pytest.mark.parametrize("changes, field", [
    ({"seed": 43}, "seed"),
    ({"token_offset": 1}, "data_sha256.train"),
    ({"num_ranks": 2}, "world_size"),
])
def test_resume_rejects_changed_run_before_loading_weights(training_entry, tmp_path, changes, field):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    path = directory / "step10" / "state.pt"
    before = path.read_bytes()
    training_entry.evaluations.clear()
    with pytest.raises(ValueError, match=field):
        training_entry.run(directory, resume_tag="step10", **changes)
    assert training_entry.engines[-1].checkpoint_loads == 0
    assert training_entry.evaluations == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("keys, value", [
    (("train_micro_batch_size_per_gpu",), 2),
    (("gradient_accumulation_steps",), 4),
    (("optimizer", "params", "lr"), 0.0002),
    (("scheduler", "params", "total_num_steps"), 40),
    (("bf16", "enabled"), False),
    (("zero_optimization", "stage"), 1),
])
def test_resume_rejects_changed_deepspeed_config(training_entry, tmp_path, keys, value):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    saved = json.loads((directory / "step10" / "resume_config.json").read_text())
    config = saved["deepspeed"]
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError, match="deepspeed\\." + "\\.".join(keys)):
        training_entry.run(directory, resume_tag="step10", ds_config=config)
    assert training_entry.engines[-1].checkpoint_loads == 0


def test_resume_rejects_changed_model_configuration(training_entry, tmp_path):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    training_entry.model_config["hidden_size"] = 256
    with pytest.raises(ValueError, match=r"model\.config\.hidden_size"):
        training_entry.run(directory, resume_tag="step10")
    assert training_entry.engines[-1].checkpoint_loads == 0


def test_resume_rejects_changed_sequence_length(training_entry, tmp_path):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    with pytest.raises(ValueError, match="seq_len"):
        training_entry.run(directory, resume_tag="step10", seq_len=64)
    assert training_entry.engines[-1].checkpoint_loads == 0


def test_legacy_checkpoint_without_config_is_rejected(training_entry, tmp_path):
    directory = tmp_path / "checkpoints"
    training_entry.run(directory, max_updates=10)
    (directory / "step10" / "resume_config.json").unlink()
    with pytest.raises(ValueError, match="resume_config.json"):
        training_entry.run(directory, resume_tag="step10")
    assert training_entry.engines[-1].checkpoint_loads == 0


def test_resume_allows_copied_checkpoint_and_equivalent_config_file(training_entry, tmp_path):
    import shutil

    original = tmp_path / "original"
    moved = tmp_path / "moved"
    first = training_entry.run(original, max_updates=10)
    shutil.copytree(original, moved)
    saved = json.loads((moved / "step10" / "resume_config.json").read_text())
    # JSON 字段顺序、checkpoint 目录和本次停止步数均可以改变。
    config = dict(reversed(list(saved["deepspeed"].items())))
    resumed = training_entry.run(moved, max_updates=20, resume_tag="step10", ds_config=config)
    assert first.global_steps == 10
    assert resumed.global_steps == 20
    assert resumed.checkpoint_loads == 1
    assert "max_updates" not in saved


def test_json_controls_deepspeed_and_loader_batch(training_entry, tmp_path):
    project_dir = Path(__file__).resolve().parents[1]
    config = json.loads((project_dir / "configs/deepspeed.json").read_text())
    config["train_batch_size"] = 8
    config["train_micro_batch_size_per_gpu"] = 2
    config.pop("gradient_accumulation_steps", None)
    config["zero_optimization"]["stage"] = 1
    config["optimizer"]["params"]["lr"] = 0.0002
    engine = training_entry.run(tmp_path / "json_source", max_updates=1, num_ranks=2, ds_config=config)
    assert engine.config == config
    assert engine.batch_sizes == [2, 2]
    assert engine.global_steps == 1
    assert engine.lr == 0.0002
