from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PocketDVDNetRecipe:
    sequence_length: int = 5
    num_color_channels: int = 3
    input_group_channels: int = 30
    stem_channels: int = 16
    down_channels_0: int = 32
    down_channels_1: int = 64
    output_mid_channels: int = 32

    @classmethod
    def from_dict(cls, recipe: dict) -> "PocketDVDNetRecipe":
        return cls(
            sequence_length=recipe.get("sequence_length", 5),
            num_color_channels=recipe.get("num_color_channels", 3),
            input_group_channels=recipe.get("input_group_channels", 30),
            stem_channels=recipe.get("stem_channels", 16),
            down_channels_0=recipe.get("down_channels_0", 32),
            down_channels_1=recipe.get("down_channels_1", 64),
            output_mid_channels=recipe.get("output_mid_channels", 32),
        )

    def to_dict(self) -> dict:
        return {
            "sequence_length": self.sequence_length,
            "num_color_channels": self.num_color_channels,
            "input_group_channels": self.input_group_channels,
            "stem_channels": self.stem_channels,
            "down_channels_0": self.down_channels_0,
            "down_channels_1": self.down_channels_1,
            "output_mid_channels": self.output_mid_channels,
        }


PUBLISHED_POCKETDVDNET_RECIPE = PocketDVDNetRecipe()
LEGACY_POCKETDVDNET_KEY_REPLACEMENTS = (
    (".inc.", ".input_block."),
    (".downc0.", ".down_0."),
    (".downc1.", ".down_1."),
    (".upc2.", ".up_1."),
    (".upc1.", ".up_0."),
    (".outc.", ".output_block."),
    (".convblock.", ".block."),
)


class PocketCvBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PocketInputCvBlock(nn.Module):
    def __init__(
        self, num_in_frames: int, num_color_channels: int, group_channels: int, out_channels: int
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                num_in_frames * num_color_channels,
                num_in_frames * group_channels,
                kernel_size=3,
                padding=1,
                groups=num_in_frames,
                bias=False,
            ),
            nn.BatchNorm2d(num_in_frames * group_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_in_frames * group_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PocketDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            PocketCvBlock(out_channels, out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PocketUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            PocketCvBlock(in_channels, in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PocketOutputCvBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PocketDenBlock(nn.Module):
    def __init__(self, recipe: PocketDVDNetRecipe):
        super().__init__()
        self.input_block = PocketInputCvBlock(
            num_in_frames=3,
            num_color_channels=recipe.num_color_channels,
            group_channels=recipe.input_group_channels,
            out_channels=recipe.stem_channels,
        )
        self.down_0 = PocketDownBlock(recipe.stem_channels, recipe.down_channels_0)
        self.down_1 = PocketDownBlock(recipe.down_channels_0, recipe.down_channels_1)
        self.up_1 = PocketUpBlock(recipe.down_channels_1, recipe.down_channels_0)
        self.up_0 = PocketUpBlock(recipe.down_channels_0, recipe.stem_channels)
        self.output_block = PocketOutputCvBlock(
            recipe.stem_channels,
            recipe.output_mid_channels,
            recipe.num_color_channels,
        )
        self.reset_params()

    @staticmethod
    def _weight_init(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def reset_params(self) -> None:
        for module in self.modules():
            self._weight_init(module)

    def forward(self, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor) -> torch.Tensor:
        x0 = self.input_block(torch.cat((in0, in1, in2), dim=1))
        x1 = self.down_0(x0)
        x2 = self.down_1(x1)
        x2 = self.up_1(x2)
        x1 = self.up_0(x1 + x2)
        residual = self.output_block(x0 + x1)
        return in1 - residual


class PocketDVDNet(nn.Module):
    def __init__(self, recipe: PocketDVDNetRecipe | dict | None = None):
        super().__init__()
        if recipe is None:
            recipe = PUBLISHED_POCKETDVDNET_RECIPE
        elif isinstance(recipe, dict):
            recipe = PocketDVDNetRecipe.from_dict(recipe)

        if recipe.sequence_length != 5:
            raise ValueError(
                f"PocketDVDNet is defined for 5-frame patches, got {recipe.sequence_length}."
            )

        self.recipe = recipe
        self.num_input_frames = recipe.sequence_length
        self.num_color_channels = recipe.num_color_channels
        self.temp1 = PocketDenBlock(recipe)
        self.temp2 = PocketDenBlock(recipe)
        self.reset_params()

    @classmethod
    def from_recipe(cls, recipe: PocketDVDNetRecipe | dict) -> "PocketDVDNet":
        return cls(recipe=recipe)

    @staticmethod
    def _weight_init(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def reset_params(self) -> None:
        for module in self.modules():
            self._weight_init(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = tuple(
            x[:, self.num_color_channels * index : self.num_color_channels * (index + 1), :, :]
            for index in range(self.num_input_frames)
        )

        if len(frames) != 5:
            raise ValueError(f"PocketDVDNet expects 5 input frames, got {len(frames)}.")

        x0, x1, x2, x3, x4 = frames
        y0 = self.temp1(x0, x1, x2)
        y1 = self.temp1(x1, x2, x3)
        y2 = self.temp1(x2, x3, x4)
        return self.temp2(y0, y1, y2)


def extract_pocketdvdnet_state_dict(checkpoint: Mapping | torch.Tensor) -> Mapping:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state", "state_dict", "params", "model"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    return checkpoint


def remap_legacy_pocketdvdnet_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    remapped_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for source, target in LEGACY_POCKETDVDNET_KEY_REPLACEMENTS:
            new_key = new_key.replace(source, target)
        remapped_state_dict[new_key] = value
    return remapped_state_dict


def load_pocketdvdnet_checkpoint(
    model: PocketDVDNet,
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
    strict: bool = True,
):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = extract_pocketdvdnet_state_dict(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported PocketDVDNet checkpoint payload: {type(state_dict)!r}")
    model.load_state_dict(remap_legacy_pocketdvdnet_state_dict(state_dict), strict=strict)
    return checkpoint


PocketDVDnet = PocketDVDNet
