from __future__ import annotations

import argparse
from copy import deepcopy


def _resolve_eval_noise_seed(config: dict, sequence_index: int, sample_index: int) -> int:
    base_seed = int(config.get("noise_seed", 1))
    noise_samples = int(config.get("noise_samples", 1))
    seed_mode = config.get("noise_seed_mode", "per_sequence")

    if seed_mode == "fixed":
        return base_seed + sample_index
    if seed_mode == "per_sequence":
        return base_seed + sequence_index * noise_samples + sample_index

    raise ValueError(f"Unsupported noise_seed_mode: {seed_mode}")


def _iter_eval_noisy_sequences(clean_sequence, config: dict, sequence_index: int):
    from dataloaders.noise import LegacyUniformNoiseModel, NoiseModel

    from .runtime import pack_frames, unpack_packed_frames

    noise_protocol = config.get("noise_protocol", "csv_realistic")
    noise_samples = int(config.get("noise_samples", 1))
    if noise_samples <= 0:
        raise ValueError("noise_samples must be positive")

    for sample_index in range(noise_samples):
        seed = _resolve_eval_noise_seed(config, sequence_index, sample_index)
        if noise_protocol == "csv_realistic":
            noise_model = NoiseModel(
                config["noise_csv"], seed=seed, num_frames=clean_sequence.shape[0]
            )
            packed_clean = pack_frames(clean_sequence.unsqueeze(0))
            noisy_packed, _ = noise_model(packed_clean)
            yield unpack_packed_frames(noisy_packed, num_frames=clean_sequence.shape[0]).squeeze(0)
            continue

        if noise_protocol == "legacy_uniform":
            noise_model = LegacyUniformNoiseModel(seed=seed, num_frames=clean_sequence.shape[0])
            noisy_sequence, _ = noise_model(clean_sequence.unsqueeze(0))
            yield noisy_sequence.squeeze(0)
            continue

        raise ValueError(f"Unsupported noise_protocol: {noise_protocol}")


def _batch_ssim(prediction, target) -> float:
    import numpy as np
    from skimage.metrics import structural_similarity

    prediction_np = prediction.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()

    total_ssim = 0.0
    for frame_index in range(prediction_np.shape[0]):
        prediction_frame = np.transpose(prediction_np[frame_index], (1, 2, 0))
        target_frame = np.transpose(target_np[frame_index], (1, 2, 0))
        total_ssim += structural_similarity(
            target_frame, prediction_frame, data_range=1.0, channel_axis=2
        )
    return float(total_ssim / prediction_np.shape[0])


def _apply_fast_override(config: dict, fast_override: bool | None = None) -> dict:
    effective_config = deepcopy(config)
    fast_enabled = bool(
        effective_config.get("fast", False) if fast_override is None else fast_override
    )
    effective_config["fast"] = fast_enabled
    if not fast_enabled:
        return effective_config

    effective_config["noise_seed_mode"] = "fixed"
    effective_config["noise_seed"] = 99
    effective_config["noise_samples"] = 1
    return effective_config


def run(config: dict) -> None:
    import time
    from pathlib import Path

    import torch
    from tqdm.auto import tqdm

    from models import PocketDVDNet, load_pocketdvdnet_checkpoint

    from .artifacts import load_recipe
    from .config import dump_json
    from .data import build_val_loader
    from .inference import denoise_sequence
    from .runtime import batch_psnr, count_parameters, select_device
    from .validate import validate_stage_inputs

    effective_config = _apply_fast_override(config)
    device = select_device(effective_config.get("device", "auto"))
    validate_stage_inputs(effective_config, stage="eval")
    output_dir = Path(effective_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe(effective_config["recipe_path"])
    model = PocketDVDNet.from_recipe(recipe).to(device)
    load_pocketdvdnet_checkpoint(model, effective_config["checkpoint_path"], map_location=device)
    model.eval()

    val_loader = build_val_loader(effective_config)
    total_psnr = 0.0
    total_ssim = 0.0
    total_noisy_psnr = 0.0
    total_runtime = 0.0
    total_frames = 0
    total_sequences = 0

    with torch.no_grad():
        for sequence_index, sequence in enumerate(tqdm(val_loader, desc="Stage 4/4 eval")):
            clean_sequence = sequence.squeeze(0).to(device)
            sequence_psnr = 0.0
            sequence_ssim = 0.0
            sequence_noisy_psnr = 0.0
            sequence_samples = 0

            for noisy_sequence in _iter_eval_noisy_sequences(
                clean_sequence, effective_config, sequence_index
            ):
                sequence_noisy_psnr += batch_psnr(noisy_sequence, clean_sequence)
                start = time.perf_counter()
                denoised_sequence = denoise_sequence(
                    model, noisy_sequence, temp_psz=effective_config.get("sequence_length", 5)
                )
                total_runtime += time.perf_counter() - start

                sequence_psnr += batch_psnr(denoised_sequence, clean_sequence)
                sequence_ssim += _batch_ssim(denoised_sequence, clean_sequence)
                total_frames += clean_sequence.shape[0]
                sequence_samples += 1

            if sequence_samples == 0:
                continue

            total_psnr += sequence_psnr / sequence_samples
            total_ssim += sequence_ssim / sequence_samples
            total_noisy_psnr += sequence_noisy_psnr / sequence_samples
            total_sequences += 1

    avg_frames_per_second = total_frames / total_runtime if total_runtime > 0 else 0.0

    metrics = {
        "avg_psnr": total_psnr / max(total_sequences, 1),
        "avg_ssim": total_ssim / max(total_sequences, 1),
        "avg_noisy_psnr": total_noisy_psnr / max(total_sequences, 1),
        "avg_frames_per_second": avg_frames_per_second,
        "num_parameters": count_parameters(model),
        "checkpoint_path": effective_config["checkpoint_path"],
        "recipe_path": effective_config["recipe_path"],
        "noise_protocol": effective_config.get("noise_protocol", "csv_realistic"),
        "noise_samples": int(effective_config.get("noise_samples", 1)),
        "noise_seed": int(effective_config.get("noise_seed", 1)),
        "noise_seed_mode": effective_config.get("noise_seed_mode", "per_sequence"),
        "fast": bool(effective_config.get("fast", False)),
        "evaluated_sequence_samples": total_sequences * int(effective_config.get("noise_samples", 1)),
        "num_validation_sequences": len(val_loader),
    }
    dump_json(output_dir / "metrics.json", metrics)


def main() -> None:
    from .config import load_config

    parser = argparse.ArgumentParser(
        description="Stage 4/4: evaluate the distilled PocketDVDNet checkpoint."
    )
    parser.add_argument(
        "--config", default="configs/paper/eval.yaml", help="Path to the stage YAML config."
    )
    fast_group = parser.add_mutually_exclusive_group()
    fast_group.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=None,
        help="Evaluate only the representative seed 99 instead of the full configured seed sweep.",
    )
    fast_group.add_argument(
        "--no-fast",
        dest="fast",
        action="store_false",
        help="Disable fast mode even if the config file enables it.",
    )
    args = parser.parse_args()
    run(_apply_fast_override(load_config(args.config), fast_override=args.fast))


if __name__ == "__main__":
    main()
