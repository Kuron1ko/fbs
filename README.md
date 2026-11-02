# DeepSpeed 双卡训练与显存诊断

项目目标与后续训练计划见 [项目指导书](deepspeed-training-project-guide.md)。当前已实现数据预处理、固定长度 token 数据集，以及单卡随机小模型的训练、评估和 checkpoint 数据位置、随机数状态恢复；正式双卡训练尚未完成。

## 环境与测试

在项目根目录使用已有虚拟环境。新环境先按指导书安装适合主机的 PyTorch，再安装 `requirements.txt`。

```bash
source .venv/bin/activate
python -m pytest -q
```

测试使用合成数据，不下载模型或数据集，不需要 GPU。

Linear + CrossEntropy 融合实验使用可选依赖 `requirements-liger.txt`。
在已准备好 PyTorch、DeepSpeed 的环境中安装并运行 CUDA 验证：

```bash
.venv/bin/python -m pip install -r requirements-liger.txt
.venv/bin/deepspeed --include localhost:0 scripts/smoke_liger.py
```

验证脚本使用小型随机 Qwen，检查 BF16 loss/梯度、共享词表权重、ZeRO-2 梯度累积和参数更新，不下载模型或写入 checkpoint。成功时输出 `LIGER_SMOKE_OK`。安装依赖不会自动启用正式训练中的融合；需要在模型构建阶段显式接入。

## 单卡 checkpoint 恢复测试

训练脚本固定读取两份文件，不再从命令行读取训练参数：

| 文件 | 内容 |
| --- | --- |
| [`configs/train.toml`](configs/train.toml) | 模型、数据、序列长度、种子、本次停止步数、评估/保存间隔、checkpoint 目录及恢复 tag |
| [`configs/deepspeed.json`](configs/deepspeed.json) | DeepSpeed 直接接受的 batch、ZeRO、精度、梯度裁剪、优化器和调度器参数 |

JSON 字典原样传给 `deepspeed.initialize`；DataLoader 也使用其中的 `train_micro_batch_size_per_gpu`。当前 JSON 中 `train_batch_size=8`、`train_micro_batch_size_per_gpu=1`，由 DeepSpeed 根据实际 GPU 数推算梯度累积次数：单卡 8 次、双卡 4 次。当前 `max_updates` 和调度器 `total_num_steps` 都为 1500；模型仍是随机小模型，验证集仍只取前 8 个样本，正式训练还需后续接入。

做短恢复测试前，先准备下文的 `data/smoke` 数据。按以下设置修改配置，使用新的或空的 checkpoint 目录：

| 文件位置 | 设置 |
| --- | --- |
| TOML `[data]` | `dir = "data/smoke"`、`seq_len = 128` |
| TOML `[train]` | `seed = 42`、`max_updates = 10`、`evaluate_interval = 5`、`save_interval = 10` |
| TOML `[checkpoint]` | `dir = "checkpoints/resume_test01"`、`resume_tag = ""` |
| JSON 顶层 | `train_batch_size = 2`、`train_micro_batch_size_per_gpu = 1`，省略 `gradient_accumulation_steps` |
| JSON `scheduler.params` | `total_num_steps = 20`、`warmup_num_steps = 4` |

在项目根目录执行：

```bash
.venv/bin/deepspeed --include localhost:0 train.py
```

程序完成 10 个参数更新步后退出，保存 `step0`、`step10`。然后只把 TOML 的 `max_updates` 改为 `20`、`resume_tag` 改为 `"step10"`，再次执行同一条命令。双卡启动命令为 `.venv/bin/deepspeed --include localhost:0,1 train.py`；`--include` 属于 DeepSpeed 启动器，设备编号通过环境变量交给引擎。

`max_updates` 是本次运行停止的累计更新步数，包含恢复前进度。调度器的 `total_num_steps` 是完整学习率计划，两次运行都保持 20。`resume_tag = ""` 表示从头训练，指定 tag 时只加载该 checkpoint，不依赖 `latest`；从头训练会拒绝覆盖非空目录。

训练数据使用 `DistributedSampler` 按 rank 分片，并按种子和 epoch 打乱。上述单卡短测累积 2 次，第 10 步结束时消费 20 个 micro-batch，第 20 步时为 40。checkpoint 保存 `epoch` 和 `batches_in_epoch`，恢复时重建 sampler 和数据迭代器，再跳过已消费 batch。恢复到已达到的目标步数时，只评估，不再训练。

每个 `step*` 目录还保存 `resume_config.json`，由 [`src/checkpoint.py`](src/checkpoint.py) 管理。恢复前核对种子、world size、模型、数据 SHA256、序列长度、sampler、DataLoader 和完整 DeepSpeed 配置；不一致时在加载权重前报错。本次停止步数和 checkpoint 目录可以改变，DeepSpeed 配置内容必须一致。数据来源还会与 `metadata.json` 中的模型/tokenizer 和数据集 ID、revision 对照，读取 token 文件时核验 SHA256。

TOML 的 `seed` 在创建模型前设置。checkpoint 保存各 rank 的 Python、NumPy、PyTorch CPU 和 CUDA 随机状态；恢复时在重建数据迭代器、跳过数据和初始评估之后读取自己的状态。CPU 测试通过临时 TOML 和 JSON 文件运行入口，覆盖配置来源、batch 和连续训练/恢复的样本及随机数序列。

每次运行的评估结果、训练样本 ID 和进度会写入 `runs/train_*.log`。CPU 测试覆盖单卡打乱后的恢复和配置不一致时的拒绝行为；真实双卡恢复仍需验证。

已完成 seed=42 的真实 GPU 对照：连续训练 20 步与“训练 10 步后重启恢复到 20 步”的最终模型状态张量逐元素一致，最大绝对差为 0；优化器和 RNG 状态也一致。完整命令、指标和日志入口见[恢复对照报告](reports/rng_resume_20261018_3d61e2.md)。

## 准备数据

首次运行需要访问 Hugging Face。脚本只下载模型的 tokenizer，不下载模型权重。
默认使用 Qwen2.5-1.5B 和 WikiText-103 raw，固定以下 revision：

- 模型：`8faed761d45a263340a0528343f099c05c9a4323`
- 数据：`b08601e04326c79dfdd32d625aee71d232d685c3`

先处理每个 split 的前 8 篇完整文章，用于本地功能检查：

```bash
python scripts/prepare_data.py --max-documents 8 --output-dir data/smoke
```

正式数据使用全部 train/validation：

```bash
python scripts/prepare_data.py --output-dir data/wikitext-103
```

`--max-documents` 限制每个 split 的编码文章数；首次仍会下载、缓存源数据文件。每批只编码少量文章，源数据使用磁盘上的 Arrow 缓存，不把全部 token 放入内存。`--batch-size` 控制每次编码的文章数，与训练 batch 无关。

输出目录必须为空，已有数据不会被覆盖。编码成功后生成：

```text
data/wikitext-103/
  train.bin
  validation.bin
  metadata.json
```

两个 `.bin` 文件均为小端 `uint32` 连续 token 流。`metadata.json` 记录模型/数据的完整 revision、文本处理规则、EOS、依赖版本、脚本 SHA256，以及每个 split 的文章数、token 数、字节数和 SHA256。使用文章上限时标记 `partial: true`，不能将这份数据作为完整 WikiText 结果。

文本规则固定为：按 WikiText 的一级标题 `= Title =` 重建文章，二级及更深标题保留在当前文章中；忽略纯空白记录；保留非空记录的空格，统一记录末尾为一个 LF；文章标题也参与编码。tokenizer 不自动添加特殊 token、不截断、不 padding，每篇文章（含末篇）末尾手工追加一个 EOS。train/validation 始终独立处理，保持原始顺序。

查看或修改输入参数：

```bash
python scripts/prepare_data.py --help
```

更换模型或数据源时同时传入对应的 `--model-revision` 或 `--dataset-revision`，必须是完整的 40 位 commit SHA；文本解析仅适用于 WikiText raw 格式。

## 读取数据

```python
from torch.utils.data import DataLoader, DistributedSampler
from src.data import TokenBlockDataset

dataset = TokenBlockDataset.from_prepared("data/smoke", "train", seq_len=128)
# 示例模拟 rank 0；正式训练从分布式进程组取得 rank/world_size。
sampler = DistributedSampler(
    dataset, num_replicas=2, rank=0, shuffle=True, seed=42, drop_last=True
)
sampler.set_epoch(0)  # 每个 epoch 开始时设置，恢复时使用原 epoch。
loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
batch = next(iter(loader))
print(batch["input_ids"].shape, batch["sample_id"])
```

`from_prepared` 默认检查元数据、文件大小和完整 SHA256，再以只读 `numpy.memmap` 读取数据。若已在运行前单独校验，可用 `verify_checksum=False` 避免每个 rank 重复扫描整个文件。也可用 `TokenBlockDataset.from_file(path, seq_len)` 直接读二进制文件，此入口不检查元数据与 SHA256。

每个样本含 `input_ids`、独立复制的 `labels` 和 `sample_id`。ID 是原始 token 流中的块索引，在同一数据文件、同一序列长度下稳定；样本覆盖 `[id * seq_len, (id + 1) * seq_len)`。末尾不足一个块的 token 不参与当前长度的训练，但仍保留在文件中。`labels` 不提前移动位置，交给 causal LM 内部处理。

后续训练调用模型时只传 `input_ids` 和 `labels`；`sample_id` 用于日志、分片及恢复检查，不传给模型。验证集后续应按 `range(rank, len(validation), world_size)` 分片，避免 sampler 补齐造成重复。

接口参考：[Datasets 加载文档](https://huggingface.co/docs/datasets/loading)、[Transformers tokenizer 文档](https://huggingface.co/docs/transformers/main_classes/tokenizer)。
