from __future__ import annotations

import argparse


def _validate(model, val_loader, config: dict, device, sequence_length: int) -> float:
    import torch

    from dataloaders.noise import NoiseModel

    from .inference import denoise_sequence
    from .runtime import batch_psnr, pack_frames, unpack_packed_frames

    model.eval()
    psnr_total = 0.0
    with torch.no_grad():
        for sequence_index, sequence in enumerate(val_loader):
            clean_sequence = sequence.squeeze(0).to(device)
            noise_model = NoiseModel(
                config["noise_csv"],
                seed=config.get("noise_seed", 1) + sequence_index,
                num_frames=clean_sequence.shape[0],
            )
            noisy_packed, _ = noise_model(pack_frames(clean_sequence.unsqueeze(0)))
            noisy_sequence = unpack_packed_frames(
                noisy_packed, num_frames=clean_sequence.shape[0]
            ).squeeze(0)
            denoised = denoise_sequence(model, noisy_sequence, temp_psz=sequence_length)
            psnr_total += batch_psnr(denoised, clean_sequence)
    return psnr_total / max(len(val_loader), 1)


def run(config: dict) -> None:
    from pathlib import Path

    import torch
    from tqdm.auto import tqdm

    from dataloaders.noise import NoiseModel
    from models import PocketDVDNet, ShiftNetTeacher

    from .artifacts import load_recipe
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
    validate_stage_inputs(config, stage="distill")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe(config["recipe_path"])
    student = PocketDVDNet.from_recipe(recipe).to(device)
    teacher = ShiftNetTeacher(
        shift_net_dir=config["external_shift_net_dir"],
        arch=config.get("teacher_arch", "gshift_denoise1"),
    ).to(device)
    teacher.load_checkpoint(config["teacher_checkpoint"], map_location=device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    train_prefetcher = build_train_prefetcher(config, return_clean_sequence=False)
    val_loader = build_val_loader(config)
    optimizer = torch.optim.Adam(student.parameters(), lr=config["lr"])
    scheduler = create_scheduler(optimizer, config["scheduler"])
    criterion = CharbonnierLoss(reduction="mean")
    scaler = create_grad_scaler(device)
    train_noise_model = NoiseModel(
        config["noise_csv"], seed=None, num_frames=config.get("sequence_length", 5)
    )
    best_psnr = float("-inf")
    alpha = config.get("distill_alpha", 0.5)
    iterations = int(config["iterations"])
    progress = tqdm(range(iterations), desc="Stage 3/4 distill")

    iteration = 0
    while iteration < iterations:
        train_prefetcher.reset()
        while iteration < iterations:
            batch = train_prefetcher.next()
            if batch is None:
                break
            clean_packed, gt_center = batch
            clean_packed = clean_packed.to(device)
            gt_center = gt_center.to(device)
            noisy_packed, added_noise = train_noise_model(clean_packed)
            noisy_sequence = unpack_packed_frames(
                noisy_packed, num_frames=config.get("sequence_length", 5)
            )
            teacher_noise_map = added_noise_to_teacher_map(
                added_noise, num_frames=config.get("sequence_length", 5)
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                with torch.no_grad():
                    teacher_center = teacher.forward_center_frame(noisy_sequence, teacher_noise_map)
                student_center = student(noisy_packed)
                teacher_loss = criterion(student_center, teacher_center)
                gt_loss = criterion(student_center, gt_center)
                loss = alpha * teacher_loss + (1.0 - alpha) * gt_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iteration += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if iteration % config.get("val_interval", 1000) == 0 or iteration == iterations:
                psnr_value = _validate(
                    student, val_loader, config, device, config.get("sequence_length", 5)
                )
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    save_checkpoint(
                        output_dir / "best.pt",
                        student,
                        optimizer,
                        scheduler,
                        iteration,
                        best_psnr,
                        extra={"recipe_path": config["recipe_path"]},
                    )
                save_checkpoint(
                    output_dir / "latest.pt",
                    student,
                    optimizer,
                    scheduler,
                    iteration,
                    best_psnr,
                    extra={"recipe_path": config["recipe_path"]},
                )
                scheduler.step()
            elif iteration % config.get("scheduler_step_interval", 1000) == 0:
                scheduler.step()

    progress.close()
    dump_json(
        output_dir / "summary.json",
        {
            "best_psnr": best_psnr,
            "best_checkpoint": str(output_dir / "best.pt"),
            "recipe_path": config["recipe_path"],
        },
    )


def main() -> None:
    from .config import load_config

    parser = argparse.ArgumentParser(
        description="Stage 3/4: distill PocketDVDNet from the retrained Shift-Net teacher."
    )
    parser.add_argument(
        "--config", default="configs/paper/distill.yaml", help="Path to the stage YAML config."
    )
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
