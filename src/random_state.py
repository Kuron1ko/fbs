import random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    """
    初始化 Python、NumPy、PyTorch 的随机种子，确保训练的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def capture_rng_state() -> dict:
    """
    保存当前的随机数状态（Python, NumPy, PyTorch CPU/GPU）。
    返回一个包含各库状态的字典。
    """
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """
    根据传入的字典，恢复 Python、NumPy、PyTorch 的随机数状态。
    """
    if "random" in state:
        random.setstate(state["random"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])