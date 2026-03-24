from __future__ import annotations

import torch
import torch.nn.functional as F


def _forward_center_frame(
    model: torch.nn.Module,
    packed_frames: torch.Tensor,
    uses_noise_map: bool,
    noise_std: float | None,
    num_color_channels: int = 3,
) -> torch.Tensor:
    if uses_noise_map:
        if noise_std is None:
            raise ValueError("noise_std is required when uses_noise_map=True")
        batch_size, packed_channels, height, width = packed_frames.shape
        num_frames = packed_channels // num_color_channels
        noise_bundle = torch.full(
            (batch_size, num_frames * num_color_channels, height, width),
            fill_value=noise_std,
            device=packed_frames.device,
            dtype=packed_frames.dtype,
        )
        return model(packed_frames, noise_bundle)
    return model(packed_frames)


def tiled_forward_center_frame(
    model: torch.nn.Module,
    packed_frames: torch.Tensor,
    uses_noise_map: bool = False,
    noise_std: float | None = None,
    num_color_channels: int = 3,
) -> torch.Tensor:
    _, _, height, width = packed_frames.shape
    pad_height = (4 - height % 4) % 4
    pad_width = (4 - width % 4) % 4
    padded = F.pad(packed_frames, (0, pad_width, 0, pad_height), mode="reflect")

    _, packed_channels, padded_height, padded_width = padded.shape
    patch_pad_h = max(16, 32 - (padded_height // 2 % 16))
    patch_pad_w = max(16, 32 - (padded_width // 2 % 16))
    output = torch.zeros(
        (1, num_color_channels, padded_height, padded_width),
        device=padded.device,
        dtype=padded.dtype,
    )

    quadrants = [
        (slice(0, padded_height // 2 + patch_pad_h), slice(0, padded_width // 2 + patch_pad_w)),
        (
            slice(0, padded_height // 2 + patch_pad_h),
            slice(padded_width // 2 - patch_pad_w, padded_width),
        ),
        (
            slice(padded_height // 2 - patch_pad_h, padded_height),
            slice(0, padded_width // 2 + patch_pad_w),
        ),
        (
            slice(padded_height // 2 - patch_pad_h, padded_height),
            slice(padded_width // 2 - patch_pad_w, padded_width),
        ),
    ]
    outputs = [
        _forward_center_frame(
            model,
            padded[:, :, h_slice, w_slice],
            uses_noise_map=uses_noise_map,
            noise_std=noise_std,
            num_color_channels=num_color_channels,
        )
        for h_slice, w_slice in quadrants
    ]

    output[:, :, : padded_height // 2, : padded_width // 2] = outputs[0][
        :, :, :-patch_pad_h, :-patch_pad_w
    ]
    output[:, :, : padded_height // 2, padded_width // 2 :] = outputs[1][
        :, :, :-patch_pad_h, patch_pad_w:
    ]
    output[:, :, padded_height // 2 :, : padded_width // 2] = outputs[2][
        :, :, patch_pad_h:, :-patch_pad_w
    ]
    output[:, :, padded_height // 2 :, padded_width // 2 :] = outputs[3][
        :, :, patch_pad_h:, patch_pad_w:
    ]

    if pad_height:
        output = output[:, :, :-pad_height, :]
    if pad_width:
        output = output[:, :, :, :-pad_width]
    return torch.clamp(output, 0.0, 1.0)


def denoise_sequence(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    temp_psz: int = 5,
    uses_noise_map: bool = False,
    noise_std: float | None = None,
    num_color_channels: int = 3,
) -> torch.Tensor:
    num_frames, channels, height, width = sequence.shape
    center_index = (temp_psz - 1) // 2
    outputs = torch.empty(
        (num_frames, channels, height, width), device=sequence.device, dtype=sequence.dtype
    )

    for frame_index in range(num_frames):
        indices = []
        for offset in range(temp_psz):
            reflected_index = frame_index + offset - center_index
            if reflected_index < 0:
                reflected_index = -reflected_index
            elif reflected_index >= num_frames:
                reflected_index = 2 * num_frames - reflected_index - 2
            indices.append(sequence[reflected_index])

        packed = torch.stack(indices, dim=0).reshape(1, temp_psz * channels, height, width)
        output_frame = tiled_forward_center_frame(
            model,
            packed,
            uses_noise_map=uses_noise_map,
            noise_std=noise_std,
            num_color_channels=num_color_channels,
        )
        outputs[frame_index] = output_frame.squeeze(0)

    return outputs
