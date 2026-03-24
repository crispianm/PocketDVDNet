from __future__ import annotations

import argparse


def _normalized_range(config: dict, key: str) -> tuple[float, float]:
    low, high = config[key]
    return low / 255.0, high / 255.0


def _validate(model, val_loader, config: dict, device) -> float:
    import torch

    from .inference import denoise_sequence
    from .runtime import batch_psnr

    model.eval()
    psnr_total = 0.0
    noise_std = config["val_noiseL"] / 255.0
    with torch.no_grad():
        for sequence in val_loader:
            clean_sequence = sequence.squeeze(0).to(device)
            noisy_sequence = torch.clamp(
                clean_sequence + noise_std * torch.randn_like(clean_sequence), 0.0, 1.0
            )
            denoised = denoise_sequence(
                model,
                noisy_sequence,
                temp_psz=config.get("sequence_length", 5),
                uses_noise_map=True,
                noise_std=noise_std,
            )
            psnr_total += batch_psnr(denoised, clean_sequence)
    return psnr_total / max(len(val_loader), 1)


def run(config: dict) -> None:
    from pathlib import Path

    import torch
    from tqdm.auto import tqdm

    from models import FastDVDNet
    from train_method.obproxsg import OBProxSG

    from .artifacts import save_recipe_artifact
    from .config import dump_json
    from .data import build_train_prefetcher, build_val_loader
    from .losses import CharbonnierLoss
    from .runtime import create_grad_scaler, model_sparsity, select_device, set_random_seed
    from .training import autocast_context, create_scheduler, save_checkpoint
    from .validate import validate_stage_inputs

    set_random_seed(config.get("seed", 1))
    device = select_device(config.get("device", "auto"))
    validate_stage_inputs(config, stage="prune")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_prefetcher = build_train_prefetcher(config, return_clean_sequence=False)
    val_loader = build_val_loader(config)
    model = FastDVDNet(num_input_frames=config.get("sequence_length", 5)).to(device)

    optimizer = OBProxSG(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["lr"],
        lambda_reg=config["optimizer"]["lambda_reg"],
        epochSize=config["optimizer"]["epoch_size"],
        Np=config["optimizer"].get("np_steps", 2),
        No=config["optimizer"].get("no_steps", "inf"),
        eps=config["optimizer"].get("eps", 1e-4),
        weight_decay=config["optimizer"].get("weight_decay", 0.0),
        lambda_warmup_steps=config["optimizer"].get("lambda_warmup_steps", 10000),
    )
    scheduler = create_scheduler(optimizer, config["scheduler"])
    criterion = CharbonnierLoss(reduction="mean")
    scaler = create_grad_scaler(device)

    noise_low, noise_high = _normalized_range(config, "noise_ival")
    best_psnr = float("-inf")
    best_metrics = {}

    iterations = int(config["iterations"])
    if iterations <= 0:
        artifact = save_recipe_artifact(config["recipe_path"], source_model=model)
        dump_json(output_dir / "summary.json", {"best_psnr": None, "recipe": artifact})
        return

    progress = tqdm(range(iterations), desc="Stage 1/4 prune")
    iteration = 0
    while iteration < iterations:
        train_prefetcher.reset()
        while iteration < iterations:
            batch = train_prefetcher.next()
            if batch is None:
                break
            clean_packed, target_frame = batch
            clean_packed = clean_packed.to(device)
            target_frame = target_frame.to(device)
            batch_size, _, height, width = clean_packed.shape

            noise_levels = torch.empty((batch_size, 1, 1, 1), device=device).uniform_(
                noise_low, noise_high
            )
            noisy_packed = clean_packed + torch.normal(
                mean=torch.zeros_like(clean_packed), std=noise_levels.expand_as(clean_packed)
            )
            noise_bundle = noise_levels.expand(
                batch_size,
                config.get("sequence_length", 5) * 3,
                height,
                width,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction = model(noisy_packed, noise_bundle)
                loss = criterion(prediction, target_frame)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iteration += 1
            progress.update(1)
            progress.set_postfix(
                loss=f"{loss.item():.4f}", sparsity=f"{model_sparsity(model):.2f}%"
            )

            if iteration % config.get("val_interval", 1000) == 0 or iteration == iterations:
                psnr_value = _validate(model, val_loader, config, device)
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    best_metrics = {"best_psnr": best_psnr, "sparsity": model_sparsity(model)}
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        iteration,
                        best_psnr,
                        extra=best_metrics,
                    )
                save_checkpoint(
                    output_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    iteration,
                    best_psnr,
                    extra=best_metrics,
                )
                scheduler.step()
            elif iteration % config.get("scheduler_step_interval", 1000) == 0:
                scheduler.step()

    progress.close()
    recipe_artifact = save_recipe_artifact(
        config["recipe_path"], source_model=model, source_checkpoint=str(output_dir / "best.pt")
    )
    best_metrics["recipe_path"] = str(config["recipe_path"])
    best_metrics["recipe_parameter_count"] = recipe_artifact["parameter_count"]
    dump_json(output_dir / "summary.json", best_metrics)


def main() -> None:
    from .config import load_config

    parser = argparse.ArgumentParser(
        description="Stage 1/4: prune FastDVDNet and emit a PocketDVDNet recipe artifact."
    )
    parser.add_argument(
        "--config", default="configs/paper/prune.yaml", help="Path to the stage YAML config."
    )
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
