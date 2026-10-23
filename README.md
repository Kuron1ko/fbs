# DeepSpeed 双卡训练与显存诊断

项目目标与后续训练计划见 [项目指导书](deepspeed-training-project-guide.md)。当前已实现数据预处理、固定长度 token 数据集，以及单卡随机小模型的训练、评估和 checkpoint 数据位置、随机数状态恢复；正式双卡训练尚未完成。

## 环境与测试

在项目根目录使用已有虚拟环境。新环境先按指导书安装适合主机的 PyTorch，再安装 `requirements.txt`。

```bash
source .venv/bin/activate
python -m pytest -q
```

测试使用合成数据，不下载模型或数据集，不需要 GPU。

## 单卡 checkpoint 恢复测试

先准备下文的 `data/smoke` 数据。在项目根目录执行，首次运行使用新的或空的 checkpoint 目录：

DeepSpeed 默认读取 [`configs/deepspeed.json`](configs/deepspeed.json)，当前设置为每卡每批 1 个样本、梯度累积 2 次、AdamW 学习率 `0.0001`。训练 DataLoader 的 batch size 也从该文件读取。可以用 `--ds-config 路径` 指定其他 JSON 文件；下面显式指定默认文件：

```bash
.venv/bin/deepspeed --include localhost:0 train.py \
  --ds-config configs/deepspeed.json \
  --seed 42 --checkpoint-dir checkpoints/resume_test01 --max-updates 10
```

程序完成 10 个参数更新步后正常退出，保存 `step0`、`step10`。重新启动，明确加载 `step10`，累计训练到 20 步：

```bash
.venv/bin/deepspeed --include localhost:0 train.py \
  --ds-config configs/deepspeed.json \
  --seed 42 --checkpoint-dir checkpoints/resume_test01 --resume-tag step10 --max-updates 20
```

`--max-updates` 表示包含恢复前进度的累计目标。`--resume-tag` 不传时从头训练，传入时只加载指定 checkpoint，不依赖 `latest`；不要通过删除 checkpoint 来控制恢复位置。从头训练会拒绝覆盖非空目录。

训练数据使用 `DistributedSampler` 按 rank 分片，并按种子和 epoch 打乱。默认配置为 `batch_size=1`、`gradient_accumulation_steps=2`，`num_workers=0`。checkpoint 的 `client_state` 保存 `epoch` 和当前 epoch 已消费的 `batches_in_epoch`；恢复时先调用 `sampler.set_epoch(epoch)` 重建顺序，再跳过这些 batch。当前 smoke 数据单卡第 10 步恢复时应记录 `epoch=0 batches_in_epoch=20`，下一批样本 ID 应与连续训练的对应位置一致；第 20 步结束时计数为 40。恢复到已达到的目标步数时，只评估，不再训练。

每个 `step*` 目录还保存 `resume_config.json`，由 [`src/checkpoint.py`](src/checkpoint.py) 管理。恢复前先校验种子、world size、模型 ID/revision 和实际模型配置、训练及验证数据 SHA256、序列长度、采样方式、DataLoader 参数和完整 DeepSpeed 配置；不一致时报告具体字段，并在 `engine.load_checkpoint()` 前停止。数据读取时仍会核对实际 token 文件的 SHA256。`--max-updates`、checkpoint 路径和配置文件路径可以改变，但 DeepSpeed 配置内容必须一致：例如分段训练 10→20 步时，调度器的 `total_num_steps` 两次都保持 20。复制 checkpoint 时应一并复制该 JSON；缺少清单的旧 checkpoint 会被拒绝，请使用新目录从头训练。

`--seed` 默认值为 42，在创建模型前设置。checkpoint 还保存各 rank 的 Python、NumPy、PyTorch CPU 和 CUDA 随机状态列表；恢复时在重建数据迭代器、跳过数据和初始评估之后读取自己的状态。CPU 测试使用 `torch.save/load` 序列化测试 checkpoint，并比较连续训练与分段恢复的样本和随机数序列，包含 epoch 中间、跨 epoch 和 epoch 末尾的情形。

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
