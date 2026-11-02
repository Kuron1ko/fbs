"""验证 Liger CUDA loss、Qwen 适配和 DeepSpeed ZeRO-2；仅使用小型随机模型。

项目根目录运行：
    .venv/bin/deepspeed --include localhost:0 scripts/smoke_liger.py
"""

from importlib.metadata import version
import json
import os

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, apply_liger_kernel_to_qwen2
from transformers import Qwen2Config, Qwen2ForCausalLM


def check_loss(device):
    """比较 BF16 融合 loss 与 PyTorch 参考计算的 loss、输入和权重梯度。"""
    torch.manual_seed(42)
    hidden = torch.randn(32, 128, device=device, dtype=torch.bfloat16).requires_grad_()
    weight = (torch.randn(1024, 128, device=device, dtype=torch.bfloat16) * 0.02).requires_grad_()
    labels = torch.randint(0, 1024, (32,), device=device)
    labels[::7] = -100
    reference = F.cross_entropy(F.linear(hidden, weight).float(), labels)
    reference_grads = torch.autograd.grad(reference, (hidden, weight))
    fused = LigerFusedLinearCrossEntropyLoss()(weight, hidden, labels)
    fused_grads = torch.autograd.grad(fused, (hidden, weight))
    torch.testing.assert_close(fused.float(), reference.float(), rtol=1e-3, atol=1e-3)
    errors = []
    for actual, expected in zip(fused_grads, reference_grads):
        assert torch.isfinite(actual).all(), "融合 loss 的梯度包含非有限值"
        error = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        assert error < 0.02, f"BF16 梯度相对 L2 误差过大：{error}"
        errors.append(error)
    assert torch.count_nonzero(fused_grads[0][labels == -100]).item() == 0
    return {"loss_abs_error": abs(fused.item() - reference.item()),
            "hidden_grad_relative_l2_error": errors[0], "weight_grad_relative_l2_error": errors[1]}


def check_qwen(device):
    """只启用 loss 融合，覆盖共享词表权重、梯度累积、参数更新及 eval 路径。"""
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=1024, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, tie_word_embeddings=True, use_cache=False,
    )
    model = Qwen2ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = input_ids.clone()
    labels[:, :3] = -100
    with torch.no_grad():
        reference_loss = model(input_ids=input_ids, labels=labels).loss.detach()
    apply_liger_kernel_to_qwen2(
        model=model, rope=False, rms_norm=False, swiglu=False,
        cross_entropy=False, fused_linear_cross_entropy=True,
    )
    assert model.lm_head.weight is model.model.embed_tokens.weight
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 4,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2, "overlap_comm": True},
            "optimizer": {"type": "AdamW", "params": {"lr": 1e-3}},
            "gradient_clipping": 1.0,
            "steps_per_print": 1000,
        },
    )
    engine.train()
    before = engine.module.lm_head.weight.detach().clone()
    losses = []
    for micro_step in range(8):
        output = engine(input_ids=input_ids, labels=labels, use_cache=False)
        assert output.logits is None, "训练路径未启用 fused linear cross entropy"
        assert torch.isfinite(output.loss), "训练 loss 包含非有限值"
        if micro_step == 0:
            torch.testing.assert_close(output.loss.float(), reference_loss.float(), rtol=1e-3, atol=1e-3)
        losses.append(output.loss.item())
        engine.backward(output.loss)
        del output
        engine.step()
    assert engine.global_steps == 2
    assert not torch.equal(before, engine.module.lm_head.weight), "优化器没有更新参数"
    engine.eval()
    with torch.no_grad():
        output = engine(input_ids=input_ids, labels=labels, use_cache=False)
        assert torch.isfinite(output.loss)
        assert output.logits.shape == (2, 32, config.vocab_size)
    return {"reference_loss": reference_loss.item(), "fused_initial_loss": losses[0],
            "optimizer_updates": engine.global_steps, "evaluation_loss": output.loss.item()}


def main():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("请用 deepspeed 启动此脚本，命令见文件顶部说明")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    assert torch.cuda.is_bf16_supported(), "此验证需要支持 BF16 的 CUDA GPU"
    deepspeed.init_distributed(dist_backend="nccl")
    try:
        results = {"versions": {name: version(name) for name in
                               ("torch", "triton", "transformers", "deepspeed", "liger-kernel")},
                   "gpu": torch.cuda.get_device_name(device),
                   "loss_check": check_loss(device), "qwen_zero2_check": check_qwen(device)}
        if dist.get_rank() == 0:
            print("LIGER_SMOKE_OK " + json.dumps(results, ensure_ascii=False), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
