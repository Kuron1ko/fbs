import torch
import torch.nn as nn
import torch.profiler

# 定义简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)

# 实例化模型和输入
model = SimpleModel().cuda()
inputs = torch.randn(64, 1024).cuda()

# 使用 torch.profiler 捕获性能数据
with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
        schedule=torch.profiler.schedule(
            wait=1,  # 前1步不采样
            warmup=1,  # 第2步作为热身，不计入结果
            active=3,  # 采集后面3步的性能数据
            repeat=2),  # 重复2轮
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profilerlog'),  # 保存日志以供 TensorBoard 可视化
        record_shapes=True,  # 记录输入张量的形状
        profile_memory=True,  # 分析内存分配
        with_stack=True  # 记录操作的调用堆栈信息
    ) as profiler:

    for step in range(10):
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()

        profiler.step()  # 更新 profiler 的步骤