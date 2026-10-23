"""记录恢复所需的配置，并在加载 DeepSpeed checkpoint 前校验。"""

import json
from pathlib import Path

import torch.distributed as dist


def build_resume_config(*, seed, model_id, revision, model, data_dir, loader, ds_config):
    """从当前模型、数据和 loader 构造可保存为 JSON 的配置快照。"""
    metadata = json.loads((Path(data_dir) / "metadata.json").read_text(encoding="utf-8"))
    config = {
        "schema_version": 1,
        "seed": seed,
        "world_size": dist.get_world_size(),
        "model": {"id": model_id, "revision": revision, "config": model.config.to_dict()},
        # from_prepared 已核对过实际文件的 SHA256；按内容识别数据，允许搬迁目录。
        "data_sha256": {split: metadata["splits"][split]["sha256"]
                        for split in ("train", "validation")},
        "seq_len": loader.dataset.seq_len,
        "sampler": {"seed": loader.sampler.seed, "shuffle": loader.sampler.shuffle,
                    "drop_last": loader.sampler.drop_last},
        "loader": {"batch_size": loader.batch_size, "drop_last": loader.drop_last,
                   "num_workers": loader.num_workers},
        "deepspeed": ds_config,
    }
    # 固定成独立的 JSON 数据，避免后续修改配置对象或 tuple/list 差异影响比较。
    # max_updates、日志目录和 checkpoint 路径不影响训练配置，不参与校验。
    return json.loads(json.dumps(config))


def _compare_config(saved, current, field=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(saved.keys() | current.keys()):
            name = f"{field}.{key}" if field else key
            if key not in saved or key not in current:
                raise ValueError(f"无法恢复训练：配置字段 {name} 缺失，请使用一致的代码和配置")
            _compare_config(saved[key], current[key], name)
    elif saved != current:
        raise ValueError(
            f"无法恢复训练：{field} 不一致（checkpoint={saved!r}，本次={current!r}）"
        )


def check_resume_config(checkpoint_dir, tag, current):
    """读取指定 tag 的配置；缺失或不一致时，在加载权重前报错。"""
    path = Path(checkpoint_dir) / tag / "resume_config.json"
    if not path.is_file():
        raise ValueError(f"checkpoint 缺少 {path}，无法验证配置；请使用新目录从头训练")
    saved = json.loads(path.read_text(encoding="utf-8"))
    _compare_config(saved, current)


def save_checkpoint(engine, checkpoint_dir, tag, client_state, resume_config):
    """保存 DeepSpeed 状态和同一 tag 下的配置清单，等待所有 rank 完成。"""
    # 所有 rank 都参与 DeepSpeed 保存；配置文件只由 rank 0 写入。
    engine.save_checkpoint(str(checkpoint_dir), tag=tag, client_state=client_state)
    if dist.get_rank() == 0:
        path = Path(checkpoint_dir) / tag / "resume_config.json"
        path.write_text(json.dumps(resume_config, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    dist.barrier()
