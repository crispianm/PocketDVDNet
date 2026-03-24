import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-4, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt((prediction - target) ** 2 + self.epsilon**2)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()
