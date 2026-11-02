"""将 MemoryRecorder 的 JSONL 日志画成 PNG。

用法：
    .venv/bin/python scripts/plot_memory.py
    .venv/bin/python scripts/plot_memory.py runs/memory_具体文件名_rank0.jsonl

默认读取最近修改的 rank 0 日志，在日志旁生成同名 PNG。
训练过程中可以重复运行；图中只包含本次读取时已完成的记录。
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # WSL / 云服务器不需要桌面窗口，也能生成图片。
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


STAGES = {
    "initialize": ("Initialize", "#64748b"),
    "evaluate": ("Evaluate", "#9333ea"),
    "save_checkpoint": ("Save checkpoint", "#15803d"),
    "load_checkpoint": ("Load checkpoint", "#be123c"),
}
METRICS = (
    "allocated_gib", "reserved_gib", "peak_allocated_gib", "peak_reserved_gib",
)


def load_records(path):
    """读取一个进程的记录；正在追加且尚未写完的末行留给下次绘图。"""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if number == len(lines) and not line.endswith("\n"):
                print("末行尚未写完，本次只画之前的完整记录。")
                break
            raise ValueError(f"{path.name} 第 {number} 行不是有效 JSON") from error
        for field in ("elapsed_s", *METRICS):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"第 {number} 行的 {field} 不是有效的非负数值")
        for field in ("stage", "update_step", "rank"):
            if field not in row:
                raise ValueError(f"第 {number} 行缺少 {field}")
        records.append(row)
    if not records:
        raise ValueError("日志中还没有完整记录，等初始化结束后再试。")
    if len({row["rank"] for row in records}) != 1:
        raise ValueError("请一次选择一个 rank 的日志，不要拼接多卡日志。")
    return records


def plot_memory(records, source):
    times = [row["elapsed_s"] for row in records]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ("#2563eb", "#ea580c")
    panels = (
        ("Current memory at interval end", METRICS[:2]),
        ("Peak memory within each interval", METRICS[2:]),
    )
    for ax, (title, fields) in zip(axes, panels):
        for field, label, color in zip(fields, ("Allocated", "Reserved (includes cache)"), colors):
            ax.plot(times, [row[field] for row in records], label=label,
                    color=color, linewidth=1.4, marker=".", markersize=3)
        for row in records:
            if row["stage"] in STAGES:
                ax.axvline(row["elapsed_s"], color=STAGES[row["stage"]][1],
                           linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("GPU memory (GiB)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="best", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    summaries = []
    for field, label, color in zip(METRICS[2:], ("Allocated", "Reserved"), colors):
        peak = max(records, key=lambda row: row[field])
        axes[1].scatter(peak["elapsed_s"], peak[field], marker="*", s=150,
                        color=color, edgecolors="white", zorder=5)
        summaries.append(
            f"{label} max: {peak[field]:.3f} GiB | {peak['stage']} | update {peak['update_step']}"
        )
        print(f"{label} 峰值：{peak[field]:.3f} GiB，阶段 {peak['stage']}，"
              f"更新步 {peak['update_step']}，记录时刻 {peak['elapsed_s']:.2f} 秒")

    seen_stages = {row["stage"] for row in records}
    handles = [Line2D([0], [0], color=color, linestyle=":", label=label)
               for stage, (label, color) in STAGES.items() if stage in seen_stages]
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4)
    fig.suptitle(f"GPU memory | rank {records[0]['rank']}\n{source.name}", fontsize=12, y=0.98)
    axes[1].set_xlabel("Elapsed time since recorder started (seconds)")
    fig.text(0.09, 0.10, "\n".join(summaries), fontsize=9)
    fig.text(0.09, 0.035,
             "Points and phase lines mark recording times, not exact times of peaks.\n"
             "Stars mark the first highest recorded values; lines connect discrete samples.",
             fontsize=8, color="#475569")
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.09, right=0.97, hspace=0.35)
    output = source.with_suffix(".png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, help="显存 JSONL；默认选最近的 rank 0 日志")
    args = parser.parse_args()
    source = args.log
    if source is None:
        runs = Path(__file__).resolve().parents[1] / "runs"
        source = max(runs.glob("memory_*_rank0.jsonl"), key=lambda p: p.stat().st_mtime, default=None)
        if source is None:
            parser.exit(1, "还没有显存日志。先启动 train.py，出现 memory_log=... 后再运行此脚本。\n")
    try:
        records = load_records(source)
        print(f"读取：{source.resolve()}（{len(records)} 条记录）")
        output = plot_memory(records, source)
    except (OSError, ValueError) as error:
        parser.exit(1, f"绘图失败：{error}\n")
    print(f"图像：{output.resolve()}")


if __name__ == "__main__":
    main()
