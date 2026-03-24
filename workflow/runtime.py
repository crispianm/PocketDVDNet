from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str | None = None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def create_grad_scaler(device: torch.device) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def pack_frames(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.dim() != 5:
        raise ValueError(f"Expected [B, F, C, H, W], got {tuple(sequence.shape)}")
    batch_size, num_frames, channels, height, width = sequence.shape
    return sequence.reshape(batch_size, num_frames * channels, height, width)


def unpack_packed_frames(packed: torch.Tensor, num_frames: int, channels: int = 3) -> torch.Tensor:
    if packed.dim() != 4:
        raise ValueError(f"Expected [B, F*C, H, W], got {tuple(packed.shape)}")
    batch_size, packed_channels, height, width = packed.shape
    expected_channels = num_frames * channels
    if packed_channels != expected_channels:
        raise ValueError(
            f"Packed tensor has {packed_channels} channels, expected {expected_channels}"
        )
    return packed.reshape(batch_size, num_frames, channels, height, width)


def added_noise_to_teacher_map(
    added_noise: torch.Tensor, num_frames: int, channels: int = 3
) -> torch.Tensor:
    noise_sequence = unpack_packed_frames(added_noise, num_frames=num_frames, channels=channels)
    return noise_sequence.abs().mean(dim=2, keepdim=True)


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    prediction = prediction.detach().float()
    target = target.detach().float()
    mse = torch.mean((prediction - target) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(max_value**2, device=mse.device) / mse))


def model_sparsity(model: torch.nn.Module) -> float:
    total_params = 0
    zero_params = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        total_params += parameter.numel()
        zero_params += torch.count_nonzero(parameter == 0).item()
    if total_params == 0:
        return 0.0
    return 100.0 * zero_params / total_params


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
