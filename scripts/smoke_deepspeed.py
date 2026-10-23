import deepspeed
import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x).float().square().mean()


assert torch.cuda.is_bf16_supported(), "当前 GPU 不支持 BF16"

model = TinyModel()
config = {
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 1,
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 2},
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
}

engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=config,
)

for step in range(2):
    x = torch.randn(2, 16, device=engine.device, dtype=torch.bfloat16)
    loss = engine(x)
    engine.backward(loss)
    engine.step()
    print(f"step={step + 1} loss={loss.item():.6f} "
          f"global_steps={engine.global_steps}", flush=True)

