from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch


class NoiseModel:
    """
    Composite real-camera noise model operating on packed video tensors.

    Inputs are expected as `[B, F*C, H, W]` in the `[0, 1]` range, where
    `F == num_frames` and `C == 3` for RGB data.
    """

    def __init__(self, dict_path: str, seed: int | None = None, num_frames: int = 5):
        self.seed = seed
        self.frames = num_frames
        self.sample_index = 0
        self.actual_labels = {
            "shot_noise": [0.0, 0.5],
            "read_noise": [0.0, 0.1],
            "uniform_noise": [0.0, 0.1],
            "row_noise": [0.0, 0.01],
            "row_noise_temp": [0.0, 0.01],
            "periodic0": [0.0, 0.5],
            "periodic1": [0.0, 0.5],
            "periodic2": [0.0, 0.5],
        }
        self.labels = pd.read_csv(dict_path)

    def _next_seed(self) -> int:
        if self.seed is None or self.seed == -1:
            return random.randint(0, 2**32 - 1)
        seed = int(self.seed) + self.sample_index
        self.sample_index += 1
        return seed

    def _scale_noise_dict(self, raw_noise_dict: dict) -> dict:
        noise_dict = {}
        for key, value_range in self.actual_labels.items():
            if key not in raw_noise_dict:
                noise_dict[key] = value_range[0]
                continue
            low, high = value_range
            noise_dict[key] = (high - low) * raw_noise_dict[key] + low
        return noise_dict

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(x)

    def apply(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"NoiseModel expects [B, F*C, H, W] input, got {tuple(x.shape)}")

        if x.min() < 0.0 or x.max() > 1.0005:
            raise ValueError("NoiseModel inputs must be normalized to [0, 1].")

        sample_seed = self._next_seed()
        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)

        raw_noise_dict = self.labels.sample(n=1, random_state=sample_seed).iloc[0].to_dict()
        noise_dict = self._scale_noise_dict(raw_noise_dict)
        batch_size, packed_channels, height, width = x.shape
        if packed_channels % self.frames != 0:
            raise ValueError(
                f"Packed channel dimension ({packed_channels}) is not divisible by num_frames ({self.frames})."
            )

        color_channels = packed_channels // self.frames
        clean_sequence = x.view(batch_size, self.frames, color_channels, height, width)
        added_noise = torch.zeros_like(clean_sequence, device=x.device)

        shot_read_factor = clean_sequence * noise_dict["shot_noise"] + noise_dict["read_noise"]
        shot_read_base = torch.randn(batch_size, 1, color_channels, height, width, device=x.device)
        added_noise += shot_read_base.expand_as(clean_sequence) * shot_read_factor

        uniform_base = torch.rand(batch_size, 1, color_channels, height, width, device=x.device)
        added_noise += noise_dict["uniform_noise"] * uniform_base.expand_as(clean_sequence)

        row_base = torch.randn(batch_size, 1, color_channels, height, 1, device=x.device)
        added_noise += noise_dict["row_noise"] * row_base.expand_as(clean_sequence)

        temporal_row_base = torch.randn(batch_size, 1, 1, 1, width, device=x.device)
        added_noise += noise_dict["row_noise_temp"] * temporal_row_base.expand_as(clean_sequence)

        periodic_fft = torch.zeros(
            batch_size,
            1,
            color_channels,
            height,
            width,
            dtype=torch.cfloat,
            device=x.device,
        )
        p0 = noise_dict["periodic0"] * torch.randn(batch_size, color_channels, device=x.device)
        p1 = noise_dict["periodic1"] * torch.randn(batch_size, color_channels, device=x.device)
        p2 = noise_dict["periodic2"] * torch.randn(batch_size, color_channels, device=x.device)
        periodic_fft[:, 0, :, 0, 0] = p0
        complex_slice = torch.complex(p1, p2)
        periodic_fft[:, 0, :, 0, width // 4] = complex_slice
        periodic_fft[:, 0, :, 0, 3 * width // 4] = torch.complex(p1, -p2)
        periodic_noise = torch.abs(torch.fft.ifft2(periodic_fft, dim=(-2, -1), norm="ortho"))
        added_noise += periodic_noise.expand_as(clean_sequence)

        noisy_sequence = torch.clamp(clean_sequence + added_noise, 0.0, 1.0)
        return noisy_sequence.view(batch_size, packed_channels, height, width), added_noise.view(
            batch_size, packed_channels, height, width
        )


class LegacyUniformNoiseModel:
    """
    Legacy evaluation-only noise model matching the older paper-time scripts.

    This model samples one uniform parameter set per call from the original
    hand-written ranges instead of sampling CSV-derived rows. It is kept
    separate from `NoiseModel` so training and the default staged evaluation
    continue to use the current realistic CSV-driven pipeline unchanged.
    """

    def __init__(self, seed: int | None = None, num_frames: int = 5):
        self.seed = seed
        self.frames = num_frames
        self.sample_index = 0
        self.param_ranges = {
            "shot_noise": (0.0, 0.5),
            "read_noise": (0.0, 0.1),
            "uniform_noise": (0.0, 0.05),
            "row_noise": (0.0, 0.01),
            "row_noise_temp": (0.0, 0.01),
            "periodic_amplitude": (0.0, 0.2),
        }

    def _next_seed(self) -> int:
        if self.seed is None or self.seed == -1:
            return random.randint(0, 2**32 - 1)
        seed = int(self.seed) + self.sample_index
        self.sample_index += 1
        return seed

    def _sample_params(self, sample_seed: int) -> dict[str, float]:
        np.random.seed(sample_seed)
        return {
            key: float(np.random.uniform(low=value_range[0], high=value_range[1]))
            for key, value_range in self.param_ranges.items()
        }

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(x)

    def apply(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.min() < 0.0 or x.max() > 1.0005:
            raise ValueError("LegacyUniformNoiseModel inputs must be normalized to [0, 1].")

        original_dim = x.dim()
        if original_dim == 4:
            batch_size, packed_channels, height, width = x.shape
            if packed_channels % self.frames != 0:
                raise ValueError(
                    f"Packed channel dimension ({packed_channels}) is not divisible by num_frames ({self.frames})."
                )
            color_channels = packed_channels // self.frames
            clean_sequence = x.view(batch_size, self.frames, color_channels, height, width)
        elif original_dim == 5:
            batch_size, num_frames, color_channels, height, width = x.shape
            self.frames = num_frames
            clean_sequence = x
        else:
            raise ValueError(
                f"LegacyUniformNoiseModel expects [B, F*C, H, W] or [B, F, C, H, W], got {tuple(x.shape)}"
            )

        sample_seed = self._next_seed()
        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)
        params = self._sample_params(sample_seed)

        noisy_sequence = clean_sequence.clone()

        shot_factor = params["shot_noise"]
        if shot_factor > 0:
            scaled_sequence = torch.clamp(clean_sequence * 255.0, min=0.0)
            poisson_noise = torch.poisson(scaled_sequence) / 255.0 - clean_sequence
            noisy_sequence += shot_factor * poisson_noise

        if params["read_noise"] > 0:
            noisy_sequence += params["read_noise"] * torch.randn_like(clean_sequence)

        if params["uniform_noise"] > 0:
            noisy_sequence += params["uniform_noise"] * (torch.rand_like(clean_sequence) - 0.5)

        if params["row_noise"] > 0:
            row_noise = torch.randn(
                batch_size,
                1,
                color_channels,
                height,
                1,
                device=clean_sequence.device,
                dtype=clean_sequence.dtype,
            )
            noisy_sequence += params["row_noise"] * row_noise

        if params["row_noise_temp"] > 0:
            temporal_row_noise = torch.randn(
                batch_size,
                1,
                1,
                1,
                width,
                device=clean_sequence.device,
                dtype=clean_sequence.dtype,
            )
            noisy_sequence += params["row_noise_temp"] * temporal_row_noise

        if params["periodic_amplitude"] > 0:
            freq_image = torch.zeros(
                batch_size,
                1,
                color_channels,
                height,
                width,
                dtype=torch.cfloat,
                device=clean_sequence.device,
            )
            amplitude = params["periodic_amplitude"]
            for frequency in (width // 4, width // 2, 3 * width // 4):
                phase = 2 * torch.pi * torch.rand(
                    batch_size, color_channels, device=clean_sequence.device
                )
                complex_amplitude = amplitude * torch.exp(1j * phase)
                freq_image[:, 0, :, 0, frequency] = complex_amplitude
                freq_image[:, 0, :, 0, -frequency] = torch.conj(complex_amplitude)
            periodic_noise = torch.fft.ifft2(freq_image, dim=(-2, -1), norm="ortho").real.abs()
            noisy_sequence += periodic_noise.expand(-1, self.frames, -1, -1, -1)

        noisy_sequence = torch.clamp(noisy_sequence, 0.0, 1.0)
        effective_noise_std = np.sqrt(
            params["shot_noise"] ** 2
            + params["read_noise"] ** 2
            + params["uniform_noise"] ** 2 / 12.0
            + params["row_noise"] ** 2
            + params["row_noise_temp"] ** 2
            + params["periodic_amplitude"] ** 2 / 2.0
        )
        noise_map = torch.full(
            (batch_size, self.frames, 1, height, width),
            float(effective_noise_std),
            device=clean_sequence.device,
            dtype=clean_sequence.dtype,
        )

        if original_dim == 4:
            return noisy_sequence.view(batch_size, self.frames * color_channels, height, width), noise_map

        return noisy_sequence, noise_map
