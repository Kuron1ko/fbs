import json
from pathlib import Path

try:
    import tomllib              # Python 3.11 及以上
except ModuleNotFoundError:
    import tomli as tomllib     # 兼容本地 Python 3.10

def load_config(path):
    with open(path, "rb") as file:
        return tomllib.load(file)


def validate_run_config(config, ds_config):
    """检查训练参数、恢复目录及预处理数据；不修改 DeepSpeed 配置。"""
    fields = [("data", "seq_len", 2), ("train", "seed", 0),
              ("train", "max_updates", 0), ("train", "log_interval", 1),
              ("train", "evaluate_interval", 1),
              ("train", "save_interval", 1)]
    for section, name, minimum in fields:
        value = config[section][name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{section}.{name} 必须是大于或等于 {minimum} 的整数")

    total_steps = ds_config.get("scheduler", {}).get("params", {}).get("total_num_steps")
    if total_steps is not None and config["train"]["max_updates"] > total_steps:
        raise ValueError("train.max_updates 不能超过 deepspeed.json 中调度器的 total_num_steps")

    profiler = config["profiler"]
    if type(profiler["enabled"]) is not bool:
        raise ValueError("profiler.enabled 必须是 true 或 false")
    if profiler["enabled"]:
        for name, minimum in (("wait", 0), ("warmup", 0), ("active", 1), ("repeat", 1)):
            if type(profiler[name]) is not int or profiler[name] < minimum:
                raise ValueError(f"profiler.{name} 必须是大于或等于 {minimum} 的整数")
        if not isinstance(profiler["dir"], str) or not profiler["dir"].strip():
            raise ValueError("profiler.dir 必须是非空路径")

    directory = Path(config["checkpoint"]["dir"])
    tag = config["checkpoint"]["resume_tag"]
    if tag:
        if not (directory / tag).is_dir():
            raise ValueError(f"找不到 checkpoint：{directory / tag}")
    elif directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("从头训练需要新的或空的 checkpoint 目录；恢复请填写 checkpoint.resume_tag")

    metadata = json.loads((Path(config["data"]["dir"]) / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "model_id": config["model"]["id"], "model_revision": config["model"]["revision"],
        "dataset_id": config["data"]["id"], "dataset_config": config["data"]["config"],
        "dataset_revision": config["data"]["revision"],
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"数据 metadata.json 的 {name} 与 train.toml 不一致")
