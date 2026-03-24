from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


class ShiftNetTeacher(nn.Module):
    """Wrapper around the external Shift-Net denoising architecture."""

    def __init__(self, shift_net_dir: str | Path, arch: str = "gshift_denoise1"):
        super().__init__()
        self.shift_net_dir = Path(shift_net_dir)
        self.arch = arch
        self.model = self._load_external_model()

    def _load_external_model(self) -> nn.Module:
        if not self.shift_net_dir.exists():
            raise FileNotFoundError(
                f"Shift-Net was not found at {self.shift_net_dir}. Run scripts/bootstrap.py first."
            )

        arch_path = self.shift_net_dir / "basicsr" / "models" / "archs" / f"{self.arch}.py"
        if not arch_path.exists():
            raise FileNotFoundError(f"Shift-Net architecture file not found: {arch_path}")

        spec = importlib.util.spec_from_file_location(f"shift_net_{self.arch}", arch_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Shift-Net module from {arch_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        options = {
            "pretrain_models_dir": str((self.shift_net_dir / "pretrained_models").resolve()) + "/"
        }
        return module.make_model(options)

    @staticmethod
    def _extract_state_dict(checkpoint: dict | torch.Tensor) -> dict:
        if isinstance(checkpoint, dict):
            for key in ("params", "model", "state_dict", "model_state"):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break
        if isinstance(checkpoint, dict) and checkpoint:
            keys = list(checkpoint.keys())
            if all(key.startswith("model.") for key in keys):
                return {key.removeprefix("model."): value for key, value in checkpoint.items()}
        return checkpoint

    def load_checkpoint(
        self, checkpoint_path: str | Path, map_location: str | torch.device | None = None
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _normalize_noise_map(noise_map: torch.Tensor, num_frames: int) -> torch.Tensor:
        if noise_map.dim() == 4:
            return noise_map.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)
        if noise_map.dim() == 5:
            return noise_map
        raise ValueError(f"Unsupported Shift-Net noise map shape: {tuple(noise_map.shape)}")

    def forward_sequence(
        self, noisy_sequence: torch.Tensor, noise_map: torch.Tensor
    ) -> torch.Tensor:
        if noisy_sequence.dim() != 5:
            raise ValueError(
                f"ShiftNetTeacher expects noisy_sequence shaped [B, F, C, H, W], got {tuple(noisy_sequence.shape)}"
            )

        batch_size, num_frames, _, _, _ = noisy_sequence.shape
        noise_map = self._normalize_noise_map(noise_map, num_frames)
        outputs = []
        for batch_index in range(batch_size):
            sequence_output = self.model(
                noisy_sequence[batch_index : batch_index + 1],
                noise_map[batch_index : batch_index + 1],
            )
            if sequence_output.dim() == 5:
                sequence_output = sequence_output.squeeze(0)
            outputs.append(sequence_output)
        return torch.stack(outputs, dim=0)

    def forward_center_frame(
        self, noisy_sequence: torch.Tensor, noise_map: torch.Tensor
    ) -> torch.Tensor:
        sequence_output = self.forward_sequence(noisy_sequence, noise_map)
        center_index = sequence_output.shape[1] // 2
        return sequence_output[:, center_index]

    def forward(self, noisy_sequence: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        return self.forward_center_frame(noisy_sequence, noise_map)


ShiftNet = ShiftNetTeacher
