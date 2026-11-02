# DeepSpeed 本地训练优化与云端验证实验指导书

更新日期：2026-10-25。

实验目的：在固定模型、数据和序列长度下，通过显存趋势和 Profiler 找出训练瓶颈，逐项改进训练实现，提高有效 tokens/s，并尽可能降低显存占用。先在本地缩小层数的模型上筛选优化，再到云端完整模型上检验收益是否仍然存在。

主线：**本地完整基线训练 → 本地短时 Profiler 诊断 → 单项优化 → 再次完整训练和短时诊断 → 云端迁移验证。**

## 0. 完成标准与边界

- **训练正确**：保留数据分片、梯度累积、学习率调度、验证、checkpoint 和恢复能力；记录训练 loss、验证 NLL/困惑度、实际更新步数。优化后重新检查正确性。
- **找到瓶颈**：记录完整运行的显存变化、峰值及其所属阶段；用短时 CPU/CUDA trace 检查数据准备、搬运、前向、反向和参数更新的耗时。
- **有公平对照**：每项优化都有原因假设、单独修改、正确性检查和复测。主要指标为不带详细 Profiler 的有效 tokens/s，显存和总耗时作为辅助指标。
- **检验迁移效果**：云端先用相同 4 层模型做短对照，再在完整 Qwen2.5-1.5B 上比较云端基线和优化版。保留无收益及退化的结果。
- **可复现**：保存 Git commit、模型/数据 revision、配置快照、初始权重或其生成方式、依赖版本、硬件信息和原始结果。

这里的“完整训练”暂定为 **1500 个参数更新步**，并不表示遍历完全部数据或达到收敛。本地随机初始化的小模型用于训练系统实验，不代表预训练模型的语言能力。云端范围为单机双卡，不扩展到多机训练。

## 1. 固定模型与基线设置

### 1.1 本地只减少层数，保留每层形状

使用 `Qwen/Qwen2.5-1.5B` 的配置，固定 revision 为 `8faed761d45a263340a0528343f099c05c9a4323`。

| 字段 | 本地模型 | 云端完整模型 |
| --- | --- | --- |
| `num_hidden_layers` | **4** | 28 |
| `hidden_size` | 1536 | 1536 |
| `intermediate_size` | 8960 | 8960 |
| `num_attention_heads` | 12 | 12 |
| `num_key_value_heads` | 2 | 2 |
| 每个 attention head 的维度 | 128 | 128 |
| `vocab_size` | 151936 | 151936 |
| `tie_word_embeddings` | `true` | `true` |
| 独立参数总数 | **420,566,528，约 0.421B** | 1,543,714,304，约 1.544B |
| 初始化 | `from_config`，种子 42 随机初始化 | `from_pretrained`，加载固定 revision 的权重 |

本地从原始配置构建模型，只把层数改为 4，其余字段保留；若配置包含逐层的 `layer_types`，也截取为前 4 项。训练统一使用 `use_cache=False`。

这样每层主要矩阵形状、GQA 结构、MLP 和词表大小都与完整模型一致，便于研究算子优化。但 4 层模型中 embedding/输出头占比更大，完整模型的端到端加速比例仍需实测。单卡结果也不能验证双卡通信收益。

### 1.2 固定训练设置

| 项目 | 本地基线 |
| --- | --- |
| 数据 | `Salesforce/wikitext`，`wikitext-103-raw-v1` |
| 数据 revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| 预处理目录 | `data/wikitext-103`，核对 `metadata.json` 和 token 文件 SHA256 |
| 序列长度 | **512 tokens** |
| 每卡 micro-batch | **2** |
| 全局 batch | **8** |
| 本地单卡梯度累积 | **4 次** |
| 随机种子 | 42 |
| 精度与 ZeRO | BF16、ZeRO-2、`overlap_comm=true` |
| 优化器 | 当前 JSON 中的 AdamW，学习率 `1e-4` |
| 调度器 | WarmupCosineLR，总步数 1500，warmup 4 步，`cos_min_ratio=0.1` |
| 完整运行 | 1500 个参数更新步，每 250 步验证，每 500 步保存 |
| 基线额外优化 | 不启用 CPU offload、activation checkpointing、Liger 或 `torch.compile` |

这轮先固定序列长度，围绕同一工作负载优化。保持优化器参数组和未显式配置的默认值一致，保存 DeepSpeed 解析后的配置。每组对照使用相同的初始权重、数据顺序和学习率计划；不要一边改性能实现，一边调整学习率或 weight decay。

当前验证只取 validation 的前 8 个样本，适合作为功能检查，报告必须标明这一范围。若要观察更可靠的验证趋势，应在正式基线开始前确定一个更大的固定子集或完整验证集，并在后续对照中保持不变。

### 1.3 本地显存预检依据

2026-10-25 在 RTX 4090 Laptop GPU（16376 MiB）上，使用上述 4 层模型和真实 WikiText token，每个配置在新进程中运行了 3 个参数更新步：

| 每卡 micro-batch | 单卡累积次数 | 峰值 allocated（GiB） | 峰值 reserved（GiB） | 结果 |
| --- | --- | --- | --- | --- |
| 1 | 8 | 9.596 | 13.504 | 通过，loss 有限 |
| **2** | **4** | **9.740** | **13.504** | **通过，作为基线** |
| 4 | 2 | 11.155 | 15.623 | 通过，留作后续增大 batch 的候选 |

详见[预检报告](reports/local_model_probe_20261025.md)与[原始数值和环境记录](reports/local_model_probe_20261025.json)。这些结果覆盖初始化和前三次更新，**没有验证完整训练、Profiler、评估和 checkpoint 保存阶段**，也不是稳态吞吐测试。

`reserved` 包含可复用的缓存，不能把它全部当成活跃张量。选择 batch 2 是保守起点；正式长跑前仍要检查各阶段的显存余量。batch 4 通过短测并不代表在所有阶段都不会 OOM。

## 2. 四阶段本地实验

### 开始前：把基线运行准备好

1. 核对第 1 节的模型参数和 JSON batch 设置，打印参数总数及实际梯度累积次数。
2. 整理 Profiler 接入：启用开关使用 `enabled`，采样步数使用 `active`；TOML 每个字段只能定义一次。用上下文管理器或配对的 `start()` / `stop()` 管理 Profiler 生命周期。
3. 用短运行覆盖参数更新、验证、保存与恢复，确认 loss 有限，恢复后的样本和学习率连续。为这次检查使用单独的输出目录。
4. 加入轻量耗时、显存采集和阶段标记。复杂实现放到 `src/` 的独立文件，`train.py` 保留少量调用。

第一阶段使用 `profiler.enabled=false`。训练入口已通过 `src/profiling.py` 接入 Profiler 上下文管理与阶段标记；第二阶段设置 `enabled=true`。单独运行 `test_profiler.py` 的简单网络只能验证 Profiler 的基本用法，实际诊断通过 `train.py` 完成。

### 阶段一：完整基线训练

- 从固定初始状态训练到 1500 个更新步，关闭详细 Profiler。
- 保留训练 loss、学习率、验证 NLL/困惑度，记录稳定训练区间的有效 tokens/s，以及包含评估和保存的总运行耗时。
- 记录显存随时间和更新步数的变化。分别观察模型/DeepSpeed 初始化、首次更新、后续训练、评估、checkpoint 保存阶段的峰值。
- 在图中标出梯度累积边界、实际参数更新、评估和保存位置，说明最高峰出现在哪个阶段。

轻量曲线用于观察长期变化。若只在每个更新步末尾采样，它不能显示该步内部的短时峰值，需要结合峰值计数和阶段二的短时诊断。

### 阶段二：短时 Profiler 诊断

另开进程和目录，从相同初始状态运行较短的训练。预热之后采集几个完整的参数更新周期，不需要为 1500 步全程开启 Profiler。

本地单卡将 `train.max_updates` 设为 15，checkpoint 使用新的诊断目录，`resume_tag` 留空；JSON 的调度总步数仍保留 1500。Profiler 配置如下：

```toml
[profiler]
enabled = true
dir = "runs/profiler"
wait = 40
warmup = 4
active = 16
repeat = 1
```

**这里的步数单位取决于 `profiler.step()` 的调用位置。** 如果像当前训练循环一样，每处理一个 micro-batch 调用一次，那么单卡累积 4 次时，上面表示等待 10 个更新周期、预热 1 个、记录 4 个，总共运行至少 60 个 micro-step，即 15 个更新周期。改变 batch 或 GPU 数后需要重新换算。完整轻量运行使用 `enabled=false`。

记录 CPU/CUDA 活动，给数据读取、CPU→GPU 搬运、前向、反向、`engine.step()` 添加阶段标记，并区分仅累积梯度和实际更新参数的步骤。CPU 标记区间不等于 CUDA 实际执行区间，要结合 GPU kernel 轨迹阅读。

优先回答：

1. GPU 空闲间隔出现在哪里，是否在等待数据、CPU 提交、同步或日志？
2. 时间主要花在 attention、MLP、输出层/交叉熵、反向还是优化器上？
3. 哪些算子或临时张量造成显存峰值，是否有不再需要但仍被引用的输出？
4. 梯度累积的普通 micro-step 与实际更新步骤有什么差别？

若短窗口没有覆盖评估或保存，就单独为这些阶段安排短诊断；不能据此断言它们没有显存问题。详细 trace、显存时间线和必要时的 snapshot 存在对应运行目录下。

每次诊断会在 `profiler.dir` 下创建独立子目录，训练日志打印 `profiler_dir=...`。采样完成后导出 `*.pt.trace.json`（算子与 CUDA 时间线）、`*.operators.txt` / `*.operators.csv`（耗时汇总）、`*.memory.html`（显存时间线）和配置快照。在项目根目录运行 `.venv/bin/tensorboard --logdir runs/profiler --host 127.0.0.1 --port 6006`，浏览器打开 `http://localhost:6006`，进入 `PYTORCH_PROFILER`，选择对应 Run，再切换到 `Trace` 视图，搜索 `train/forward` 等 section 名称。需要安装 `tensorboard` 和 `torch-tb-profiler`。也可以直接在浏览器中打开 memory HTML。这些结果用于定位瓶颈，正式吞吐仍以关闭详细 Profiler 的运行衡量。

### 阶段三：逐项优化

按“观测证据 → 原因假设 → 一项修改 → 正确性检查 → 性能复测”执行。先处理 trace 中占比较大或造成空闲的部分，不需要把第 5 节的技巧全部实现。

- 每项优化先与基线比较 loss、有效 token 数及梯度/参数更新结果，事先确定适合 BF16 的数值误差容限；计算顺序变化时不要求逐位相同。
- 初始对照固定 micro-batch 2、全局 batch 8、序列长度 512、输入顺序和初始权重。
- 每个配置使用新进程；普通短基准可预热 10 个更新步、测量 30–50 步、重复 3 次。若有编译或其他尚未稳定的开销，应延长预热并单独记录首次编译成本。
- 正式吞吐和常规显存表来自关闭详细 Profiler 的运行。诊断开销单独记录。
- 单项验证后才组合有效的修改，再测组合是否相互影响。

如果优化释放了显存，可以额外比较 micro-batch **2→4**、单卡累积 **4→2**，保持全局 batch 8。把这组实验单独列出，区分实现优化本身和增大 batch 的收益。先验证较长短测、评估、保存及 Profiler 阶段的容量，再运行完整训练。

### 阶段四：完整复测与再次诊断

选定最终优化组合后，重复阶段一的 1500 步完整运行和阶段二的短时诊断。重新从相同初始状态开始，不从基线结束时的 checkpoint 接着训练。

对照训练/验证曲线、有效 tokens/s、总耗时、峰值 allocated/reserved、峰值阶段和 GPU 空闲间隔。解释收益来源，并检查瓶颈是否转移。保留无收益、退化和 OOM 的结果；未完成的运行不计入正常吞吐均值。

## 3. 配置、启动与恢复

### 3.1 TOML 与 JSON 的职责

| 文件 | 内容 |
| --- | --- |
| `configs/train.toml` | 模型和数据身份、数据目录、序列长度、种子、停止更新步数、评估/保存间隔、checkpoint 目录与恢复 tag、Profiler 设置 |
| `configs/deepspeed.json` | DeepSpeed 能直接读取的 batch、精度、ZeRO、优化器、调度器、梯度裁剪等配置 |

DataLoader 的 batch size 从 JSON 的 `train_micro_batch_size_per_gpu` 读取，不在 TOML 再定义一份。JSON 指定全局 batch 和每卡 micro-batch 后，由 DeepSpeed 根据进程数推导累积次数。

基线 JSON：

```json
{
  "train_batch_size": 8,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_clipping": 1.0,
  "bf16": {"enabled": true},
  "zero_optimization": {"stage": 2, "overlap_comm": true},
  "optimizer": {"type": "AdamW", "params": {"lr": 0.0001}},
  "scheduler": {
    "type": "WarmupCosineLR",
    "params": {
      "total_num_steps": 1500,
      "warmup_num_steps": 4,
      "warmup_type": "linear",
      "warmup_min_ratio": 0.0,
      "cos_min_ratio": 0.1
    }
  }
}
```

公式：**全局 batch = 每卡 micro-batch × 累积次数 × GPU 数**。上述配置单卡累积 4 次，双卡累积 2 次。

### 3.2 启动与独立运行

完成第 2 节的准备后，在项目根目录启动；训练参数从配置文件读入，设备选择仍由 DeepSpeed 启动器负责：

```bash
# 本地单卡
.venv/bin/deepspeed --include localhost:0 train.py

# 云端同机双卡
.venv/bin/deepspeed --include localhost:0,1 train.py
```

每组实验设置独立的 checkpoint 和结果目录。从头运行时，`checkpoint.resume_tag` 留空，`checkpoint.dir` 使用新的或空的目录。恢复时填写原目录和实际存在的 tag，例如 `step500`；所有 rank 参与 DeepSpeed 保存和加载，普通汇总报告由 rank 0 写入。

`train.max_updates` 是本次停止的累计更新步数；JSON 的 `total_num_steps` 是完整学习率计划。短诊断可以较早停止，但仍保留 1500 步的调度计划，以便与完整运行同一步的学习率一致。需要短程恢复测试时，保存间隔也要覆盖计划退出的位置。

保留恢复配置一致性检查。更改模型、batch 或优化配置的对照应使用新运行，不要放宽检查后恢复旧实验。跨 GPU 数恢复不属于本轮验证范围。

训练核心仍是 engine 的 forward、`engine.backward(loss)`、`engine.step()`。不要额外手动调用优化器更新、重复做 causal-label shift，或使用 `device_map="auto"` 提前跨卡分配模型。

## 4. 测量指标与口径

| 指标 | 记录方式与解释 |
| --- | --- |
| 有效 tokens/s | 计入 loss 的 token 数除以耗时；当前无 padding、无忽略标签的因果语言模型，每个长度 S 的样本为 S−1 个有效 token。多卡汇总全局 token 数。 |
| 稳态耗时 | 测量完整更新周期，包含该周期的所有累积 micro-step。在计时区间边界同步 CUDA，避免只测到异步提交时间。 |
| 总运行耗时 | 单独报告包含初始化、评估和保存的完整运行耗时；与稳态训练吞吐分开。 |
| allocated / reserved | 分别记录当前值和峰值；说明重置峰值的时机及覆盖阶段。多卡保留每卡数据，并取每卡峰值中的最大值。 |
| 显存趋势与峰值位置 | 时间或更新步为横轴，标注阶段；步末当前值、区间峰值和全程累计峰值分别标明。 |
| loss / NLL / 困惑度 | 固定输入、学习率、样本顺序及验证范围，检查数值正确性和训练趋势。 |
| GPU 利用率 | 用来辅助解释空闲和负载变化，最终是否有效以吞吐、耗时和正确性为准。 |

轻量采集避免每个算子都同步。详细 Profiler 的 shape/stack/memory 记录可能增加耗时并影响张量生命周期，因此它的峰值与速度要标为诊断结果。

PyTorch allocator 统计和 snapshot 主要覆盖 PyTorch 管理的内存，NCCL 等外部分配可能不包含在内。结合设备级占用采样，但注明采样间隔可能错过短峰值。参阅与运行版本对应的 [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) 和[显存可视化文档](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)。

## 5. 可选优化与练习资料

### 提高效率，可能同时减少显存

| 候选 | 何时值得尝试 | 需要检查 |
| --- | --- | --- |
| Liger 的 RMSNorm、SwiGLU、融合输出层与交叉熵 | 相应算子或大词表 logits 在 trace 中开销明显 | Qwen/软件版本支持、loss reduction、标签移位、梯度与参数更新正确性 |
| 模型或模块级 `torch.compile` | CPU 提交或许多小算子占比较高 | 编译成本、graph break、动态形状、DeepSpeed 兼容性；先选局部模块验证 |
| pinned memory、非阻塞搬运、调整数据加载 | GPU 在等待输入或 CPU→GPU 搬运 | 是否实际减少等待；增加 worker 未必有益，还要检查内存和恢复后的数据顺序 |
| 调整日志与标量读取频率 | trace 显示日志、磁盘写入或 GPU 标量同步造成等待 | 保留必要统计和故障信息；固定并报告测量时的日志策略 |

### 主要减少显存

| 候选 | 原理 | 代价或限制 |
| --- | --- | --- |
| Activation checkpointing | 少保存中间激活，反向时重新计算 | 通常增加计算；需要比较节省的显存与吞吐变化 |
| 释放不再需要的 outputs/logits 引用 | 避免上一轮输出延长张量生命周期 | 只能在反向和必要统计完成后释放，先确认引用确实造成额外占用 |
| 减小 ZeRO 通信 bucket | 降低部分临时缓冲需求 | 可能增加通信次数；云端双卡必须重新测，不能只按单卡结果选择 |

当前 BF16、关闭 KV cache、预检中使用的 SDPA 路径，以及本地这套 DeepSpeed AdamW 配置选用的 FusedAdam，属于已有基线能力。不要把它们重复记作新增优化；具体 attention CUDA 内核仍要从 trace 确认。

资料：[Liger 支持的模型与算子](https://github.com/linkedin/Liger-Kernel)、[PyTorch 调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)、[pinned memory 与非阻塞搬运](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。

算子级在线练习可使用 [LeetGPU](https://leetgpu.com/challenges)、[Tensara](https://tensara.org/) 和 [GPU MODE](https://www.gpumode.com/)。这些平台帮助练习算子正确性和速度，项目收益仍需在实际模型、软件版本和 GPU 上验证。

## 6. 云端迁移验证

本地筛出有效修改后再开展云端实验。先检查代码、依赖、数据和配置是否对应；已归档的本地 PyTorch 为 `2.14.1+cu132`，云端为 `2.6.0+cu126`，两者不同。尽量对齐关键软件版本；暂时不能对齐时，分别在各环境内部做基线/优化对照，并把版本差异列入迁移结论的限制。

1. **同模型检查**：在云端使用与本地相同的 4 层模型和输入形状，运行基线/优化版短对照，观察换硬件后的效果。
2. **完整模型检查**：切换到固定 revision 的 28 层预训练 Qwen2.5-1.5B，在同一台双卡机器上建立完整模型自己的基线，再应用同一优化。
3. **先验证容量与正确性**：每卡 micro-batch 2、全局 batch 8 时，双卡累积 2 次。先短测更新、评估和保存；如果需要降低 batch 或启用重计算，应给云端基线和优化版共同采用新的可运行设置，并单独记录。
4. **重复稳态测量**：比较有效 tokens/s、整步耗时、相关算子耗时及每卡峰值显存。必要时再做短时 Profiler，检查瓶颈是否转移。
5. **按预算补长跑**：迁移判断至少要有正确性检查和重复的稳定短测；若要声称长期训练质量或稳定性不变，再补相同预算的完整训练对照。

本地与云端各自计算相对基线的收益，不直接把两台机器的绝对速度差当成优化收益。重点回答“哪些优化保留了收益、哪些失效、原因是什么”，不预设一定能迁移。单卡无法决定云端通信重叠和 bucket 的最佳设置；4 层随机模型也无法替代完整预训练模型的质量检查。

## 7. 文件职责与最终交付

保留现有训练结构，复杂功能写到独立文件再由 `train.py` 调用。下表含已有文件和后续建议新增的文件，列出职责不表示已经全部实现。

| 文件 | 职责 |
| --- | --- |
| `README.md` | 环境、准备数据、单卡/双卡命令、恢复方法和结果入口 |
| `src/config.py` | 读取 TOML，校验训练设置及其与数据元信息、JSON 的关系 |
| `src/model.py` | 构建 4 层本地模型或加载完整预训练模型，管理模型侧优化开关 |
| `scripts/prepare_data.py`、`src/data.py` | 固定 revision 预处理、token block 与稳定样本 ID；保持数据和顺序可复现 |
| `train.py` | 组织数据、engine、训练、评估、保存、恢复，调用测量工具 |
| `src/evaluate.py` | 无重复分担验证样本，汇总 token 加权 NLL 和有效 token 数 |
| `src/checkpoint.py`、`src/random_state.py` | 保存/恢复训练与 RNG 状态，检查恢复配置一致性 |
| `src/metrics.py`（待补） | 轻量耗时、有效 tokens/s、显存和实验元数据记录 |
| `src/memory_probe.py`（待补） | Profiler 启停、阶段标记、显存时间线和必要时的 snapshot |
| `scripts/bench_training.sh`（按需新增） | 每个配置及重复启动新进程，保留配置快照、输出目录和失败日志 |
| `scripts/report.py`、`reports/report.md`（待补） | 汇总训练曲线、显存曲线、阶段耗时和优化前后对照 |
| `tests/` | 保留数据分片、batch、评估、恢复测试；新增优化需有针对性的正确性验证 |

最终报告至少包含：

| 部分 | 交付内容 |
| --- | --- |
| 实验身份 | 模型参数、初始状态、数据 SHA256、配置、版本、硬件、commit |
| 本地完整基线 | 1500 步训练/验证曲线、显存趋势、峰值阶段、吞吐及总耗时 |
| 瓶颈证据 | 代表性 trace、阶段/算子耗时、显存事件和原因假设 |
| 单项优化与组合 | 修改内容、正确性检查、重复测量、相对收益及退化案例 |
| 完整复测 | 与基线同口径的完整训练和短时诊断对照 |
| 云端迁移 | 云端自己的基线/优化结果，哪些收益保留及可能原因 |
| 可复现性 | 启动步骤、原始结果位置、限制、云端运行时间和实际费用 |

## 8. 环境、同步与运行管理

### 本地与云端环境

本地在 WSL2 Ubuntu 的 `~/fbs` 中运行，性能测试避免把项目和数据放在 `/mnt/c` 等 Windows 挂载目录。使用 Windows 侧 NVIDIA 驱动，不在 WSL 中安装 Linux 显卡驱动，参阅 [CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)。

云端已选配置：同一台机器的 **2 × RTX 4090 24 GB**，镜像 **PyTorch 2.6.0 + CUDA 12.6 + cuDNN 9 + Python 3.11 + Ubuntu 22.04**，磁盘 **100 GB**。确认主机内存和可用磁盘能容纳模型、数据及计划保留的 checkpoint。

新环境先检查：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv
python3 --version
nvcc --version
```

在目标机器建立自己的虚拟环境。按验证过的环境记录安装 PyTorch 和项目依赖；需要选择新 wheel 时使用 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。不要复制本地 `.venv` 到云端。记录 PyTorch/DeepSpeed/Transformers 版本、CUDA、驱动、编译器、`pip freeze` 和 `ds_report`，保存在 `reports/environment/`。

DeepSpeed 算子可能首次即时编译。编译失败时检查 toolkit、PyTorch CUDA 版本和编译器，参考 [DeepSpeed 安装说明](https://www.deepspeed.ai/tutorials/advanced-install/)。正式计时前完成编译和必要预热。

### 同步、备份与长任务

代码、配置和小型报告进入 Git；数据、权重、checkpoint、大型 trace 和日志单独备份，不提交密钥。每次实验保存配置快照，不只依赖之后可能被修改的 `configs/`。

```bash
git status --short
# 检查差异后，显式添加这次需要同步的文件
git add deepspeed-training-project-guide.md
git commit -m "Update training optimization experiment guide"
git push origin main
git rev-parse HEAD
```

首次在云端 clone 仓库；后续先 `git fetch origin`，再检出对应 commit，例如 `git switch --detach <commit>`。长任务用 `tmux`，实验运行中保持代码和配置不变。

上传预处理数据后核对元数据及 token 文件 SHA256。停止实例前确认日志已落盘，备份结果和必要 checkpoint，记录配置、commit 与实际用时。按实例时价、磁盘及流量费用预算；先完成本地筛选，再集中安排云端对照。

## 9. 对照实验重点避免的问题

- 把 micro-step 当成参数更新步，导致预热、测量、保存和 Profiler 窗口错位。
- 同时改变 batch、学习率、模型或多个优化，无法解释收益来源。
- 从基线训练结束的权重开始优化版，导致训练起点不同。
- 把 Profiler 或首次编译开销混入常规吞吐结果。
- 把 reserved 全部当成活跃张量，或只用低频设备采样判断峰值。
- 只追求 GPU 利用率百分比，没有检查有效 tokens/s 和训练正确性。
- 直接把 4 层模型的整步收益比例外推到 28 层模型。
- 用单卡结果判断双卡通信效果，或忽略软件版本差异。
- 只保存成功结果，遗漏退化、OOM 和仍无法解释的现象。
- 把三个更新步的容量预检当成完整训练和所有附加阶段都已验证。
