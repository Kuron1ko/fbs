**云端双卡冒烟与恢复验收：通过。**

2026-10-24，在本地 CPU 上读取云端备份，对比连续训练 20 个更新步与训练 10 步、退出、恢复到 20 步的结果。机器可读结果见 [cloud_resume_20261024.json](cloud_resume_20261024.json)。

原始备份为 Windows 下载目录中的 `cloud-smoke-results.tar.gz`，大小 1,138,924,526 字节。本地抽取的日志、两个最终 checkpoint 和环境记录位于 `runs/cloud_audit_20261024/`，该目录被 Git 忽略；原始备份需要另行保留。核对脚本为 [verify.py](../runs/cloud_audit_20261024/verify.py)，从仓库根目录执行 `.venv/bin/python runs/cloud_audit_20261024/verify.py`，本次退出码为 0。

环境记录显示：两张 NVIDIA GeForce RTX 4090（每卡 24564 MiB），Python 3.11.15、PyTorch 2.6.0+cu126、CUDA toolkit 12.6、DeepSpeed 0.19.7、Transformers 5.20.0。归档记录的云端 commit 为 `510aa18114463f57e52dc44b76221d1694c53bd2`；本次没有独立核验云端工作区是否与该 commit 完全一致。

本轮使用缩小配置、随机初始化的 Qwen2 模型：2 层、hidden size 128、intermediate size 512。序列长度 128、种子 42、每卡 micro-batch 1、梯度累积 2、world size 2，因此全局 batch 为 4。采用 BF16、ZeRO-2、AdamW 和总计 20 步的 WarmupCosineLR。模型配置来源虽然记录为 Qwen2.5-1.5B，本轮并未训练完整的预训练 1.5B 模型。

| 核对内容 | 结果 |
| --- | --- |
| 启动及退出 | 连续、前半段、恢复三次运行均有两个进程成功退出，日志无 Traceback 或 `[ERROR]` |
| 样本顺序 | 每个 rank 的连续运行 40 个 micro-batch，与前后两段拼接后的记录完全一致 |
| 跨 rank 重复 | 本轮两 rank 各 40 个样本，rank 内无重复，rank 间交集为空 |
| 恢复位置 | 从更新步 10、epoch 0、已消费 20 个 micro-batch 恢复；下一批样本 ID 分别为 rank 0 的 98、rank 1 的 103 |
| 最终进度 | 两组均为更新步 20、epoch 0、每 rank 消费 40 个 micro-batch，累计全局样本数 80 |
| 模型权重 | `module` 的 27 个状态张量逐元素一致，最大绝对差为 0；权重均为有限数 |
| 优化器 | 两个 rank 的优化器 checkpoint 均一致，各含 3 个张量，张量最大绝对差为 0 |
| 调度器 | 保存状态一致，`last_batch_iteration=19` |
| 随机数状态 | 每个 rank 的 Python、NumPy、PyTorch 和 CUDA RNG 状态均一致 |
| 其他恢复状态 | 完整模型 checkpoint 与 `resume_config.json` 的递归比较均通过 |

比较张量时同时检查 shape、dtype 和 `torch.equal`；没有采用近似容差。优化器中的 `LossScaler` 对象比较类型与保存的属性，两组均为 `cur_scale=1.0, dynamic=False`，不比较反序列化后对象的内存地址。

下列验证指标在两组运行对应更新步的日志中一致；loss 和困惑度的比较精度限于日志打印的六位小数。每次验证计入 1016 个预测 token。`lr_next` 是该更新步结束后记录的学习率。

| 更新步 | 验证 NLL | 困惑度 | lr_next |
| --- | --- | --- | --- |
| 0 | 11.958870 | 156196.491304 | 未记录 |
| 5 | 11.939894 | 153260.401237 | 9.9135338e-05 |
| 10 | 11.882143 | 144660.245841 | 7.2220754e-05 |
| 15 | 11.846563 | 139603.684523 | 2.999934e-05 |
| 20 | 11.833185 | 137748.582191 | 1e-05 |

这次验收覆盖相同环境、相同双卡数量下的小模型训练与恢复，且只运行了 epoch 0 的部分样本。完整预训练模型、跨 epoch 恢复、完整验证集、长训练稳定性以及显存和吞吐实验仍需分别验证。本次未测量训练峰值显存，不将归档中的空闲 GPU 快照作为训练显存结果。
