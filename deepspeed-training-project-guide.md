# DeepSpeed 双卡训练与显存诊断项目指导书

目标：在 14 天内完成一个可复现的双卡训练项目。第一部分按常规方法完成一次真正的全参数继续预训练；第二部分自己构建显存诊断与容量预测工具，回答固定硬件上能训练多长序列、为什么某个配置会 OOM。后者是工程探索，不宣称提出新的 ZeRO 算法。

## 0. 完成标准与边界

- **标准训练主线**：同一台 Linux 云主机上的 2 张同型号 GPU，使用 DeepSpeed、BF16、ZeRO-2、梯度累积、学习率调度、分布式数据采样、验证集评估、checkpoint 与恢复。训练前后验证集 NLL/困惑度有记录；若没有改善，要分析并修正训练设置，不能只展示吞吐。
- **特色专题**：自己写显存采集、归因和预测代码。用少量序列长度校准，预测未参与校准的长度是否能运行、峰值显存大致是多少，然后实测检验。给出显存与有效 tokens/s 的 Pareto 图，以及失败案例。
- **可复现**：报告模型与数据 revision、Git commit、配置、依赖版本、GPU 型号、原始 JSONL 日志和一条从新云实例启动训练的命令。
- **边界**：只做单机双卡数据并行，不声称完成多机训练；不重写 ZeRO、NCCL、优化器或异步 checkpoint 引擎；不把短程继续预训练的结果说成通用模型能力提升。

建议目标模型为 [Qwen2.5-1.5B Base](https://huggingface.co/Qwen/Qwen2.5-1.5B)，数据为 [WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext) 的官方 train/validation split。前者是 1.54B 参数的开源基础模型。若租到的 2 x 24 GB GPU 无法在序列长度 512、每卡 micro-batch 1 下稳定训练，就先记录 OOM 证据，再启用 activation checkpointing；仍无法跑通才将主线缩到 0.6B 模型，并在报告中明确说明。不要悄悄更换模型后把结果放在同一张表中。

## 1. 两周路线

| 时间 | 工作 | 当天结束时应有的结果 |
| --- | --- | --- |
| 第 1-2 天 | 学 DeepSpeed 训练 API、ZeRO 与 batch 公式；装本地 WSL2 环境 | 单 GPU 小模型 2 步冒烟运行 |
| 第 3-4 天 | 数据预处理、固定 token block、数据分片测试 | 数据 SHA256、样本 ID 无重复测试 |
| 第 5-6 天 | 标准训练循环、分布式评估、checkpoint/恢复 | 本地小模型训练、验证、恢复均通过 |
| 第 7 天 | 租同机双卡，部署同一 Git commit 与依赖 | 云端双卡 20 步冒烟通过 |
| 第 8-9 天 | 训练主线与验证、日志、一次恢复测试 | 主线运行完成，训练曲线与硬件信息齐全 |
| 第 10-11 天 | 显存采集、校准与留出长度预测 | 可运行/OOM 预测和峰值显存对照表 |
| 第 12 天 | 对异常预测做定位，补测 ZeRO-3 或 checkpointing 对照 | 一张 Pareto 图和至少一条有证据的结论 |
| 第 13-14 天 | 整理报告、复现命令、简历描述 | 新环境可照 README 复现，代码和报告可公开 |

**租卡门槛**：第 6 天之前不要为了写代码一直开着云 GPU。本地使用小模型或缩小层数、序列长度测试代码路径；性能数字只从云端同一台双卡机器采集。

## 2. 需要先读的资料

按使用顺序读，不必通读整个框架源码：

1. [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)：`deepspeed.initialize()`、engine 的 forward/backward/step、单机启动、保存与恢复。
2. [DeepSpeed 配置文档](https://www.deepspeed.ai/docs/config-json/)：BF16、AdamW、调度器和 `global batch = micro-batch x accumulation x GPU 数`。
3. [ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/)：各阶段到底切分哪些训练状态。
4. [PyTorch DistributedSampler](https://docs.pytorch.org/docs/stable/data.html)：rank 分片、`set_epoch()`、`drop_last`。
5. 专题开始时读 [PyTorch 显存快照](https://docs.pytorch.org/docs/stable/torch_cuda_memory) 和 [DeepSpeed Profiler](https://www.deepspeed.ai/tutorials/pytorch-profiler/)。显存快照默认只看到 PyTorch allocator 管理的分配，NCCL 等外部分配需要另作估计。

## 3. 硬件、环境与依赖

### 本地：Windows + 4090 Laptop

在管理员 PowerShell 中安装 WSL2 Ubuntu（已有则跳过）：

```powershell
wsl --install -d Ubuntu
wsl --update
wsl
```

安装或更新 **Windows 侧** NVIDIA 驱动。不要在 WSL 里安装 Linux 显卡驱动；按 [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)检查。项目放在 WSL 的 `~/ds-train-lab`，不要放在 `/mnt/c` 或 `/mnt/d` 上做正式性能测试。

在 WSL Ubuntu 中：

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip build-essential ninja-build tmux
nvidia-smi
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel ninja
```

先到 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 Linux、Pip、与你的驱动兼容的 CUDA wheel，把生成的命令在虚拟环境中执行；不要照抄某篇旧教程的 CUDA 版本。然后安装项目其余依赖：

```bash
python -m pip install deepspeed "transformers>=4.37" datasets numpy pandas matplotlib pytest
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
ds_report
```

开发初期先把上述非 PyTorch 依赖写入 `requirements.txt`；第一次云端冒烟通过后，把 DeepSpeed、Transformers、Datasets、NumPy 等顶层包的**实际验证版本**固定进去。把 PyTorch wheel 安装命令、驱动、`nvcc --version` 和 `pip freeze` 记录在 `reports/environment/`。本地与云端要用同一份代码和相容的包版本，但驱动及 CUDA toolkit 属于各自主机，**不要复制虚拟环境目录**。DeepSpeed 算子可能首次即时编译；如失败，先检查 `ds_report`、`nvcc`、PyTorch CUDA 版本及编译器，不要靠忽略 CUDA 检查来掩盖问题。[安装说明](https://www.deepspeed.ai/tutorials/advanced-install/)

### 云端：一台双卡虚拟机

选择同一台机器上的 2 张同型号、各至少 24 GB 显存的 GPU；建议至少 64 GB 内存、100 GB 可持久化存储。选择包含 PyTorch 和 CUDA toolkit 的 Ubuntu 镜像，确认能 SSH 登录。不要把两台单卡虚拟机当作本项目的“双卡”：那会额外引入多机网络和存储问题。

首次登录后先检查，不要急着下载模型：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv
python3 --version
nvcc --version
```

在云端 clone Git 仓库，建立新的 `.venv`，按与本地相同的顺序安装 PyTorch 和 `requirements.txt`，再运行 CUDA 检查与 `ds_report`。如果云镜像已经提供了可工作的 PyTorch，优先记录并沿用它，不必先升级。模型文件在云端单独下载；预处理 token 数据可在云端按相同 revision 重新生成，或从本地上传后核对 SHA256。

**磁盘规则**：代码、少量结果在 Git；数据、模型权重、运行日志、checkpoint 不在 Git。确认云平台停止/释放实例后哪些盘仍保留；即使有持久盘，也在停机前把报告、原始日志和必要 checkpoint 备份到本地。长训练在 `tmux` 中运行。

## 4. 仓库结构与逐文件职责

以下是你将创建的 Git 仓库，不是让你从头实现 DeepSpeed：

```text
ds-train-lab/
  README.md
  .gitignore
  requirements.txt
  configs/
    train.toml
    deepspeed.json
  train.py
  src/
    __init__.py
    data.py
    model.py
    evaluate.py
    metrics.py
    memory_probe.py
    memory_model.py
  scripts/
    prepare_data.py
    smoke.sh
    train_2gpu.sh
    bench_memory.sh
    report.py
    collect_env.sh
  tests/
    test_data.py
    test_batch.py
    test_resume.py
    test_memory_model.py
  reports/
    report.md
    environment/
  data/                 # .gitignore
  runs/                 # .gitignore
  checkpoints/          # .gitignore
```

| 文件 | 要写什么；怎样判断写对了 |
| --- | --- |
| `README.md` | 项目目标、硬件要求、环境搭建、准备数据、单卡/双卡命令、恢复命令、结果入口。新实例照做能跑 20 步。 |
| `.gitignore` | 忽略 `.venv/`、`data/`、`runs/`、`checkpoints/`、模型权重和密钥；保留 `configs/`、代码与 `reports/`。 |
| `requirements.txt` | 固定验证过的顶层 Python 包版本；PyTorch 的 CUDA wheel 单独按环境说明安装。 |
| `configs/train.toml` | 模型/数据 ID 与完整 revision SHA、`seq_len=512`、全局 batch 8、每卡 micro-batch 1、1500 个参数更新步、随机种子、评估和保存间隔。revision 可由 Hugging Face 页面或 `huggingface_hub` 的 `model_info(...).sha` / `dataset_info(...).sha` 查询；若调整数值必须记入报告。 |
| `configs/deepspeed.json` | BF16、AdamW、WarmupCosineLR、梯度裁剪、ZeRO-2；先不启用 CPU/NVMe offload。`train.py` 根据 GPU 数补齐 batch 与累积次数，并保存解析后的配置。 |
| `scripts/prepare_data.py` | 单进程下载固定 revision 的 WikiText-103 raw，使用模型 tokenizer 编码，文档间插入 EOS，写 `train.bin` / `validation.bin`（`uint32`）、元数据、SHA256。空行处理必须确定且可重复。 |
| `src/data.py` | 从二进制 token 流构造固定长度、不重叠的训练块与稳定样本 ID；训练集 `DistributedSampler(drop_last=True)` 按 rank 分片，epoch 开始调用 `set_epoch()`。初版 DataLoader 用 `num_workers=0` 降低恢复顺序的复杂度。 |
| `src/model.py` | 加载固定 revision 的基础模型，训练时关闭 `use_cache`；提供 activation checkpointing 开关和 AdamW 参数组（norm/bias 不施加 weight decay）。不要使用 `device_map="auto"`，由 DeepSpeed 接管设备。 |
| `train.py` | 参数解析、校验 batch 公式、初始化 DeepSpeed、训练循环、周期验证、所有 rank 参与 checkpoint、恢复 epoch/批次位置及 RNG 状态。只在**参数更新边界**计数与保存，不把 micro-step 当 update。 |
| `src/evaluate.py` | 各 rank 用 `range(rank, len(validation), world_size)` 无重复地分担验证集，累计 token 加权 NLL 并 `all_reduce`；汇总困惑度。评估不可只平均每个 rank 的 loss 均值，也不要让默认 sampler 为凑齐 rank 数而复制尾部样本。 |
| `src/metrics.py` | 每轮记录全局更新步、loss、lr、tokens/s、step 耗时、显存和 GPU/配置元数据为 JSONL；rank 0 写汇总，rank 级数据单独保留。 |
| `src/memory_probe.py` | 预热后同步 CUDA，重置峰值统计，记录每卡 `max_memory_allocated` 与 `max_memory_reserved`。只在短诊断运行中生成显存快照，不把 profiler 开着跑主基准。 |
| `src/memory_model.py` | 基于参数数目、ZeRO 阶段和少量实测点，建立简单、单调的峰值显存估计；给出预测值、误差及安全余量。只宣称对本模型/硬件附近有效。 |
| `scripts/bench_memory.sh` | 每个 `(seq_len, micro_batch, ZeRO, checkpointing)` 配置启动**新的训练进程**，做预热、短测和 OOM 记录，避免前一次运行的 allocator 状态污染下一次。 |
| `scripts/report.py`、`reports/report.md` | 从 JSONL 生成训练曲线、显存预测误差图与显存-吞吐 Pareto 图；写明负结果、局限及完整复现命令。 |
| `tests/` | 测跨 rank 样本不重复、batch 公式、数据恢复后下一个样本 ID、显存模型对未见点的预测流程。测试默认不需要租卡。 |

`configs/deepspeed.json` 可以从下面这个最小配置开始；`train.py` 把 `train_batch_size`、`train_micro_batch_size_per_gpu`、`gradient_accumulation_steps` 填入运行时配置。DeepSpeed 支持 AdamW 和 WarmupCosineLR，参数字段见[配置文档](https://www.deepspeed.ai/docs/config-json/)及[调度器文档](https://deepspeed.readthedocs.io/en/stable/schedulers.html)。

```json
{
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "zero_optimization": {"stage": 2, "contiguous_gradients": true},
  "optimizer": {
    "type": "AdamW",
    "params": {"lr": 0.00001, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}
  },
  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {"total_num_steps": 1500, "warmup_min_ratio": 0.0,
               "warmup_num_steps": 100, "cos_min_ratio": 0.1}
  }
}
```

训练核心流程应是 `outputs = engine(input_ids=..., labels=...)`、`engine.backward(outputs.loss)`、`engine.step()`；不要再手工调用 `loss.backward()` 或重复做 causal-label shift。若只指定全局 batch 8、micro-batch 1，单卡累积 8 次，双卡累积 4 次。训练与评估数据、初始化权重、学习率等不能因为比较卡数而悄悄改变。

## 5. 开发和运行顺序

1. **先写数据与测试**：准备少量合成 token 流，断言固定块数量、索引范围、两个 rank 无重复；再处理真实 WikiText。正式数据的模型 revision、数据 revision、token 数和 SHA256 写进 `data/metadata.json`。
2. **本地冒烟**：用 0.6B 模型或缩小配置、序列长度 128、2 个更新步测试 forward/backward、验证和 checkpoint。恢复时重建同一 `DistributedSampler`、调用相同的 `set_epoch()`，跳过已提交的 micro-batch，比较恢复前后的下一批样本 ID。你的 Laptop 4090 的显存、功耗与云端卡不同，本地结果不进入正式性能表。
3. **云端双卡冒烟**：从同一 commit 运行 20 个更新步；检查两个 rank 都启动、数据样本 ID 不重叠、loss 有限、`engine.global_steps` 与预期一致、两卡有显存占用。
4. **主线训练**：在固定 `seq_len=512` 下先评估初始验证集，再运行 1500 个更新步，每 250 步评估、每 500 步保存；中途做一次退出再恢复，检查后续样本 ID 与 step 连续。根据实测速度可缩放步数，但应保持足够 token 数、完整记录调整理由。
5. **专题实验**：保留主线训练结果不动，另启短基准。先在长度 512/1024/2048 上校准显存估计，再预测 1536/4096 等未参与校准的长度；若某配置已 OOM，就以更小的长度替换校准点。每个配置预热 10 个更新步、测 30-50 步、重复 3 次。
6. **解释异常**：比较 PyTorch 已分配/已保留显存与 `nvidia-smi` 总占用；必要时对一轮失败配置抓 snapshot。不要把 allocator 快照看不到的 NCCL/驱动内存硬归因到模型激活。检查 ZeRO-2/3、activation checkpointing 的取舍，但重点是预测与解释，非盲目扫开关。

示例启动命令；实际参数名由你在 `argparse` 中按此契约实现：

```bash
# 单卡冒烟
deepspeed --include localhost:0 train.py --run-config configs/train.toml --ds-config configs/deepspeed.json --max-steps 20

# 同机双卡主线；单机启动不需要 hostfile
deepspeed --include localhost:0,1 train.py --run-config configs/train.toml --ds-config configs/deepspeed.json

# 恢复：路径指向完整的 DeepSpeed checkpoint 根目录
deepspeed --include localhost:0,1 train.py --run-config configs/train.toml --ds-config configs/deepspeed.json --resume checkpoints/<run-id>
```

DeepSpeed 单机 launcher 的 `--include localhost:0,1` 写法见[官方入门](https://www.deepspeed.ai/getting-started/)。`train.py` 要接受 launcher 传入的 `local_rank` 或读取 `LOCAL_RANK`。所有 rank 都要调用 DeepSpeed 的保存接口；只有 rank 0 写普通汇总日志。完整的 checkpoint 包含模型、优化器、调度器与自定义 `client_state`。恢复时至少核对模型、数据、配置和 world size 与原运行一致；本项目**不做改变 GPU 数后的弹性恢复**。

## 6. 专题：显存预算下的长序列决策

这个专题的交付不是“试了几种配置，挑最快”，因为 [DeepSpeed Autotuner](https://www.deepspeed.ai/tutorials/autotuning/)已能自动搜索 ZeRO 阶段和 micro-batch。你的贡献必须包含**归因 + 事先预测 + 留出配置验证**。

1. 记录模型参数数目，并用 ZeRO 阶段解释权重、梯度与优化器状态是复制还是分片；把这部分作为理论固定开销的粗估。不要假设 DeepSpeed bucket、混合精度副本和临时缓冲的实际字节数恰好等于教科书公式。
2. 在同一台双卡机器上测 `seq_len` 与 micro-batch 对峰值显存的影响。分别拟合 ZeRO-2、ZeRO-3 及 checkpointing 开/关的**局部**估计；先试 `M_peak = c0 + c1 x (micro_batch x seq_len)` 并约束 `c1 >= 0`。若留出点残差很大，先检查 attention 实现、临时缓冲与 allocator 行为是否变化，不为追求拟合优度任意堆参数。
3. 在测留出长度之前，写下预测：是否能跑、峰值区间、最可能的 OOM 来源。随后才启动留出实验；报告实际值、误差和失败原因。至少保留一个预测不准的案例，不删异常值。
4. 最终的建议是条件性的，例如“在这台 2 x 24 GB 主机、该模型、固定优化器下，长度 X 选择 ZeRO-2 + checkpointing 比 ZeRO-3 更省时间；超过 Y 则不成立”。用显存余量、有效 tokens/s、验证 loss 的正确性检查共同支持，不把不同上下文长度的 tokens/s 当作同一种训练任务的唯一排名。

主线质量训练和专题短基准要分开：基准不需要跑到收敛，主线不能因 profiler 开销或试验性 OOM 反复中断。每个正式基准记录 GPU、功耗限制、commit SHA、数据 SHA、模型 revision、完整 DeepSpeed 配置与是否启用 activation checkpointing。

## 7. 本地与云端同步流程

本地 WSL 中创建 Git 仓库，连接你自己的 GitHub/GitLab 私有仓库。不要把 SSH 私钥或 Hugging Face token 提交进去。

```bash
# 本地：每个可运行里程碑提交
git status --short
git add README.md configs train.py src scripts tests requirements.txt .gitignore
git commit -m "Implement two-GPU DeepSpeed training"
git push origin main
git rev-parse HEAD
```

云端首次 `git clone <repo-url> ~/ds-train-lab`。后续正式实验先 `git fetch origin`，再 `git switch --detach <本地记录的commit SHA>`；实验运行中不 `git pull`。若临时在云端改代码，要提交并推回远端，再让本地同步，避免两份代码分叉。长任务可在云端执行 `tmux new -s train` 后运行 `bash scripts/train_2gpu.sh`；按 `Ctrl-b`、`d` 离开会话，重新登录后用 `tmux attach -t train` 查看。

若本地已完成数据预处理，在本地 WSL 上传小型元数据及 token 文件；大模型权重建议在云端按固定 revision 下载。把以下占位符替换为实际 SSH 用户、主机、端口与云端**绝对路径**：

```bash
scp -P <SSH_PORT> -r data user@host:<REMOTE_PROJECT_ABSOLUTE_PATH>/
scp -P <SSH_PORT> -r user@host:<REMOTE_PROJECT_ABSOLUTE_PATH>/runs ./runs-cloud
scp -P <SSH_PORT> -r user@host:<REMOTE_PROJECT_ABSOLUTE_PATH>/reports ./reports-cloud
```

传输后核对 `data/metadata.json` 和 token 文件 SHA256。checkpoint 较大，至少保留最近一个可恢复版本到持久盘，并在释放实例前备份需要交付的版本。结果分析完，把小型 CSV/图和总结写进 Git 仓库 `reports/`，原始大日志仍放云端/本地备份。

## 8. 结果表、报告与简历材料

`reports/report.md` 至少回答下列问题：

| 部分 | 必须呈现 |
| --- | --- |
| 训练有效性 | 初始/最终验证 NLL、训练曲线、数据规模、更新步数；区分短程继续预训练和通用能力评测 |
| 分布式正确性 | 两个 rank 的样本 ID 检查、batch 公式、一次 checkpoint 恢复验证 |
| 系统表现 | GPU 型号、BF16/ZeRO 配置、平均及波动 tokens/s、峰值显存、计时方法 |
| 特色专题 | 校准点、事先预测、留出点实测、误差、OOM 归因、显存-吞吐图、反例 |
| 可复现性 | Git commit、模型/数据 revision、依赖、完整命令、原始日志位置、云端成本 |

不要承诺一定得到“ZeRO-3 更快”或“预测误差低于 10%”。有证据的负结果同样有价值。若专题最终只是调了几个配置、没有预测和解释，它就不能算本项目的突出部分，应在最后两天优先补上诊断与留出实验。

## 9. 费用与停机规则

只在云端双卡冒烟、本次训练、短基准和必要补测时计费；按**双卡实例时价 x 实际开机小时数 + 持久盘/流量费用**预算，并留 20%-30% 缓冲。先在本地完成代码和测试，尽量把云端使用压在约 10-20 个实例小时内；这是工作量预算，不是价格保证。每次结束：确认日志写盘、备份结果、记录 commit 与配置、再停机。不同平台对停机后磁盘和实例保留的规则不同，租用前以该平台当前文档为准。

## 10. 最常见的错误

- **双卡训练两张卡读相同数据**：测试每个 rank 前 100 个样本 ID，确认互不重叠；使用 `DistributedSampler` 并正确设置 epoch。
- **把 micro-step 当作 optimizer step**：梯度累积下，`engine.step()` 可被每个 micro-batch 调用，但参数更新只发生在累积边界；计时、日志和保存要以实际更新步为准。
- **比较时改变多个变量**：质量对比固定数据与全局 batch；显存专题明确一次只改变序列长度、micro-batch 或 ZeRO/重计算中的少数因素。
- **只看 `nvidia-smi` 的采样峰值**：它可能错过短时峰值；同时记录 PyTorch peak allocated/reserved，并说明 allocator 外内存的限制。
- **本地 Laptop 与云端卡直接比快慢**：散热、功耗、显存和软件环境不同；本地只做功能测试，正式结论全在同一云主机上。
- **`device_map="auto"` 与 DeepSpeed 混用**：模型不要提前由 Transformers 跨卡放置，交给 DeepSpeed 初始化。
- **ZeRO-3 直接从模型 `state_dict` 保存普通权重**：用 DeepSpeed checkpoint API；若要导出普通权重，再使用其提供的转换工具。

