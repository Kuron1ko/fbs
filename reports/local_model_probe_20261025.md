**本地实验模型建议：保留 Qwen2.5-1.5B 的宽度和算子结构，只将层数从 28 降到 4。**

2026-10-25 在 NVIDIA GeForce RTX 4090 Laptop GPU（16376 MiB）上，用独立脚本完成每卡 micro-batch 为 1、2、4 的显存预检。原始数值、环境和 DeepSpeed 配置快照见 [JSON 记录](local_model_probe_20261025.json)。本次没有修改生产训练入口或模型构建函数。

| 模型字段 | 本地候选 | 云端目标 |
| --- | --- | --- |
| `num_hidden_layers` | 4 | 28 |
| `hidden_size` | 1536 | 1536 |
| `intermediate_size` | 8960 | 8960 |
| `num_attention_heads` | 12 | 12 |
| `num_key_value_heads` | 2 | 2 |
| head dimension | 128 | 128 |
| `vocab_size` | 151936 | 151936 |
| `tie_word_embeddings` | true | true |
| 独立参数总数 | 420,566,528 | 1,543,714,304 |

配置源固定为 `Qwen/Qwen2.5-1.5B@8faed761d45a263340a0528343f099c05c9a4323`。候选模型通过 `from_config` 随机初始化，种子 42，`use_cache=False`，`layer_types` 同步截取前 4 层。本地候选不具有完整预训练模型的语言能力。

每层主要矩阵形状与云端目标一致，有利于筛选算子优化。完整模型与 4 层模型的 embedding/输出头大小相同，但它们占总计算和参数量的比例不同，所以端到端加速比例不能直接外推。单卡不能验证双卡的通信、分片和计算重叠效果。

各候选深度的参数预算如下。20 字节/参数来自本地 DeepSpeed ZeRO-2 单卡、无卸载估算器，包含其模型状态和梯度临时转换预算；它不是含激活、运行时缓冲和 allocator 缓存的实测总峰值。

| 层数 | 参数数 | ZeRO-2 状态估算（GiB） |
| --- | --- | --- |
| 2 | 326,970,880 | 6.09 |
| 4 | 420,566,528 | 7.83 |
| 6 | 514,162,176 | 9.58 |
| 8 | 607,757,824 | 11.32 |
| 28 | 1,543,714,304 | 28.75 |

参考：[DeepSpeed 显存估算](https://deepspeed.readthedocs.io/en/stable/memory.html)。

实测固定序列长度 512、全局 batch 8、BF16、ZeRO-2、AdamW，无 CPU offload，无 activation checkpointing。采用当前 JSON 中的其他 DeepSpeed 设置，包括 `overlap_comm=true`，没有人工缩小 bucket。解析到的 attention backend 为 SDPA；本次未录制 kernel trace 来判定具体 CUDA attention 内核。

| 每卡 micro-batch | 累积次数 | 峰值 allocated（GiB） | 峰值 reserved（GiB） | 结果 |
| --- | --- | --- | --- | --- |
| 1 | 8 | 9.596 | 13.504 | 3 个参数更新步通过，loss 有限 |
| 2 | 4 | 9.740 | 13.504 | 3 个参数更新步通过，loss 有限 |
| 4 | 2 | 11.155 | 15.623 | 3 个参数更新步通过，loss 有限 |

每个配置在新进程中执行；峰值计数在 DeepSpeed 初始化前重置，覆盖首次优化器更新和后续两个更新。输入来自已有的 WikiText token 文件。保留了当前训练循环中输出对象的生命周期，没有额外删除输出以降低测试显存。

推荐先以 **micro-batch 2、累积 4 次、全局 batch 8** 建立基线。batch 4 已通过普通短测，但应在较长短测以及 Profiler、评估、保存等实际阶段验证后，再用于完整运行。reserved 包含可复用缓存，并不等于同等大小的活跃张量；不能仅凭 reserved 数字断定会 OOM。这里没有寻找显存极限，也没有保证任何配置永不 OOM。

本次未验证完整 1500 步、Profiler 额外开销、评估和 checkpoint 保存阶段。三个更新步没有经过规范稳态计时，不把 JSON 中的 wall time 用作吞吐排名。

复现预检需保留忽略目录中的 [脚本](../runs/local_model_probe/probe.py)。从项目根目录运行，例如：

```bash
PROBE_MICRO_BATCH=2 HF_HUB_OFFLINE=1 .venv/bin/deepspeed --include localhost:0 --master_port 29641 runs/local_model_probe/probe.py
```

本地优化先以相同 seed、数据、序列长度、每卡 batch 和全局 batch 比较 baseline/优化版；先检查 loss 和梯度/更新结果，再测不带详细 Profiler 的有效 tokens/s 与显存。需要调整 micro-batch 时另开一组对照，保持全局 batch 一致。每个技巧单独测试，最终组合再测。

云端先用相同 4 层模型和同样输入形状做短对照，再用完整 28 层模型验证。尽量对齐 PyTorch、Transformers 和 DeepSpeed 版本；当前本地 PyTorch 2.14.1+cu132 与已归档云端 2.6.0+cu126 不一致。最终报告分别记录单个算子/模块耗时和整步耗时，保留无收益或退化的结果。

可优先研究的非默认优化：Liger 的 RMSNorm/SwiGLU/融合输出层与交叉熵、模型或模块级 `torch.compile`、根据 trace 调整数据搬运与日志频率。显存方向包括 activation checkpointing、释放不再需要的 logits 引用、显式调整 ZeRO bucket；最后一项涉及双卡通信取舍，不能仅在单卡上选定云端值。BF16、当前的 SDPA 路径，以及本地 DeepSpeed 这套 AdamW 配置默认选择的 FusedAdam，均属于已有基线能力。

资料：[Liger 支持的模型及算子](https://github.com/linkedin/Liger-Kernel)、[PyTorch 调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)、[异步搬运说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。

算子级在线练习可使用 [LeetGPU](https://leetgpu.com/challenges)、[Tensara](https://tensara.org/) 和 [GPU MODE](https://www.gpumode.com/)。这些平台可以辅助验证算子正确性和速度，不能替代本项目在实际模型、软件版本和 GPU 上的端到端训练测试。

训练入口目前还有独立的接入问题：TOML 的 `[profiler]` 重复定义 `active`，且训练文件使用了 `prOfiler`/`prefiler` 两种错误拼写，并缺少 Profiler 启停管理。独立预检不经过这段代码，因此预检通过不代表当前 `train.py` 可以直接启动。
