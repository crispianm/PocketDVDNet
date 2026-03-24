from __future__ import annotations

import argparse


def _validate(model, val_loader, config: dict, device) -> float:
    import torch

    from dataloaders.noise import NoiseModel

    from .runtime import added_noise_to_teacher_map, batch_psnr, pack_frames, unpack_packed_frames

    model.eval()
    psnr_total = 0.0
    with torch.no_grad():
        for sequence_index, sequence in enumerate(val_loader):
            clean_sequence = sequence.to(device)
            num_frames = clean_sequence.shape[1]
            clean_packed = pack_frames(clean_sequence)
            noise_model = NoiseModel(
                config["noise_csv"],
                seed=config.get("noise_seed", 1) + sequence_index,
                num_frames=num_frames,
            )
            noisy_packed, added_noise = noise_model(clean_packed)
            noisy_sequence = unpack_packed_frames(noisy_packed, num_frames=num_frames)
            teacher_noise_map = added_noise_to_teacher_map(added_noise, num_frames=num_frames)
            denoised_sequence = model.forward_sequence(noisy_sequence, teacher_noise_map)
            psnr_total += batch_psnr(denoised_sequence.squeeze(0), clean_sequence.squeeze(0))
    return psnr_total / max(len(val_loader), 1)


def run(config: dict) -> None:
    from pathlib import Path

    import torch
    from tqdm.auto import tqdm

    from dataloaders.noise import NoiseModel
    from models import ShiftNetTeacher

    from .config import dump_json
    from .data import build_train_prefetcher, build_val_loader
    from .losses import CharbonnierLoss
    from .runtime import (
        added_noise_to_teacher_map,
        create_grad_scaler,
        select_device,
        set_random_seed,
        unpack_packed_frames,
    )
    from .training import autocast_context, create_scheduler, save_checkpoint
    from .validate import validate_stage_inputs

    set_random_seed(config.get("seed", 1))
    device = select_device(config.get("device", "auto"))
    validate_stage_inputs(config, stage="teacher")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_prefetcher = build_train_prefetcher(config, return_clean_sequence=True)
    val_loader = build_val_loader(config)
    teacher = ShiftNetTeacher(
        shift_net_dir=config["external_shift_net_dir"],
        arch=config.get("teacher_arch", "gshift_denoise1"),
    ).to(device)

    pretrained_checkpoint = config.get("pretrained_checkpoint")
    if pretrained_checkpoint and Path(pretrained_checkpoint).exists():
        teacher.load_checkpoint(pretrained_checkpoint, map_location=device)

    optimizer = torch.optim.Adam(teacher.parameters(), lr=config["lr"])
    scheduler = create_scheduler(optimizer, config["scheduler"])
    criterion = CharbonnierLoss(reduction="mean")
    scaler = create_grad_scaler(device)
    train_noise_model = NoiseModel(k
        config["noise_csv"], seed=None, num_frames=config.get("sequence_length", 5)
    )
    best_psnr = float("-inf")
    iterations = int(config["iterations"])
    progress = tqdm(range(iterations), desc="Stage 2/4 teacher")

    iteration = 0
    while iteration < iterations:
        train_prefetcher.reset()
        while iteration < iterations:
            batch = train_prefetcher.next()
            if batch is None:
                break
            clean_packed, _, clean_sequence = batch
            clean_packed = clean_packed.to(device)
            clean_sequence = clean_sequence.to(device)
            noisy_packed, added_noise = train_noise_model(clean_packed)
            noisy_sequence = unpack_packed_frames(
                noisy_packed, num_frames=config.get("sequence_length", 5)
            )
            teacher_noise_map = added_noise_to_teacher_map(
                added_noise, num_frames=config.get("sequence_length", 5)
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                denoised_sequence = teacher.forward_sequence(noisy_sequence, teacher_noise_map)
                loss = criterion(denoised_sequence, clean_sequence)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iteration += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if iteration % config.get("val_interval", 1000) == 0 or iteration == iterations:
                psnr_value = _validate(teacher, val_loader, config, device)
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    save_checkpoint(
                        output_dir / "best.pt", teacher, optimizer, scheduler, iteration, best_psnr
                    )
                save_checkpoint(
                    output_dir / "latest.pt", teacher, optimizer, scheduler, iteration, best_psnr
                )
                scheduler.step()
            elif iteration % config.get("scheduler_step_interval", 1000) == 0:
                scheduler.step()

    progress.close()
    dump_json(
        output_dir / "summary.json",
        {"best_psnr": best_psnr, "best_checkpoint": str(output_dir / "best.pt")},
    )


def main() -> None:
    from .config import load_config

    parser = argparse.ArgumentParser(
        description="Stage 2/4: fine-tune Shift-Net on the realistic paper noise model."
    )
    parser.add_argument(
        "--config", default="configs/paper/teacher.yaml", help="Path to the stage YAML config."
    )
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
