"""短时训练诊断：Profiler 生命周期、阶段标记与独立结果目录。"""

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import csv
import json
import uuid

import torch


class TrainingProfiler:
    def __init__(self, run_config, ds_config, *, rank, device, logger):
        self.run_config = run_config
        self.ds_config = ds_config
        self.config = run_config["profiler"]
        self.enabled = self.config["enabled"]
        self.rank = rank
        self.device = torch.device(device)
        self.logger = logger
        self.output_dir = None
        self.profiler = None

    def __enter__(self):
        if not self.enabled:
            return self

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.output_dir = Path(self.config["dir"]) / f"profile_{run_id}_rank{self.rank}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        snapshot = {
            "run_config": self.run_config, "deepspeed": self.ds_config,
            "rank": self.rank, "device": str(self.device),
            "torch_version": torch.__version__, "schedule_unit": "micro_step",
        }
        (self.output_dir / "config.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.config["wait"], warmup=self.config["warmup"],
                active=self.config["active"], repeat=self.config["repeat"],
            ),
            on_trace_ready=self._export,
            record_shapes=True, profile_memory=True, with_stack=True,
        )
        self.logger.info("profiler_dir=%s", self.output_dir)
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiler is not None:
            return self.profiler.__exit__(exc_type, exc_value, traceback)
        return False

    def section(self, name):
        # 基线关闭 Profiler 时，不创建 record_function 标记。
        return torch.profiler.record_function(name) if self.enabled else nullcontext()

    def engine_step(self, engine):
        if not self.enabled:
            return nullcontext()
        # 必须在 engine.step() 前判断当前 micro-step 是否会更新参数。
        name = "train/optimizer_step" if engine.is_gradient_accumulation_boundary() else "train/accumulate_step"
        return self.section(name)

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def _export(self, profiler):
        stem = self.output_dir / f"micro_step{profiler.step_num}"
        # TensorBoard 的 PyTorch Profiler 插件按 .pt.trace.json 后缀识别结果。
        profiler.export_chrome_trace(str(stem) + ".pt.trace.json")
        events = profiler.key_averages(group_by_input_shape=True)
        sort_by = "self_device_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
        (Path(str(stem) + ".operators.txt")).write_text(
            events.table(sort_by=sort_by, row_limit=30) + "\n", encoding="utf-8"
        )
        # 保存完整汇总，便于后续分析，而不只保留屏幕上的前 30 行。
        with Path(str(stem) + ".operators.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("operator", "input_shapes", "calls", "self_cpu_time_us",
                             "self_device_time_us", "self_device_memory_bytes"))
            for event in events:
                writer.writerow((event.key, event.input_shapes, event.count, event.self_cpu_time_total,
                                 event.self_device_time_total, event.self_device_memory_usage))
        profiler.export_memory_timeline(str(stem) + ".memory.html", device=str(self.device))
        self.logger.info("profiler_saved=%s", stem)
