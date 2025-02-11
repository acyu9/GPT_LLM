import torch

def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device