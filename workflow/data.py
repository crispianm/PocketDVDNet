from __future__ import annotations

import torch

from dataloaders.fastdvdnet import DVDDataset, ValDataset
from utils.prefetcher import CPUPrefetcher, PrefetchDataLoader


def build_train_prefetcher(config: dict, return_clean_sequence: bool = False) -> CPUPrefetcher:
    dataset = DVDDataset(
        root_dir=config["trainset_dir"],
        sequence_length=config.get("sequence_length", 5),
        ctrl_fr_idx=config.get("sequence_length", 5) // 2,
        crop_size=config["patch_size"],
        return_clean_sequence=return_clean_sequence,
    )
    loader = PrefetchDataLoader(
        config.get("num_prefetch_queue", 8),
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    return CPUPrefetcher(loader)


def build_val_loader(config: dict) -> torch.utils.data.DataLoader:
    dataset = ValDataset(
        valsetdir=config["valset_dir"],
        gray_mode=False,
        num_input_frames=config.get("val_sequence_length", 25),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=config.get("val_num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
    )
