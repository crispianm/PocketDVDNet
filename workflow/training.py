from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR

from utils.lr_scheduler import CosineAnnealingRestartLR, MultiStepRestartLR


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict):
    scheduler_type = scheduler_config["name"]
    if scheduler_type in {"MultiStepLR", "MultiStepRestartLR"}:
        return MultiStepRestartLR(
            optimizer,
            milestones=scheduler_config["milestones"],
            gamma=scheduler_config["gamma"],
        )
    if scheduler_type == "CosineAnnealingRestartLR":
        return CosineAnnealingRestartLR(
            optimizer,
            periods=scheduler_config["periods"],
            restart_weights=scheduler_config["restart_weights"],
            eta_min=scheduler_config["eta_min"],
        )
    return StepLR(
        optimizer,
        step_size=scheduler_config.get("step_size", 30),
        gamma=scheduler_config.get("gamma", 0.5),
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    iteration: int,
    best_psnr: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "iteration": iteration,
        "best_psnr": best_psnr,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location=None,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
