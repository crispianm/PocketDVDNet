from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloaders.fastdvdnet import ValDataset
from dataloaders.noise import NoiseModel
from models import PocketDVDNet, ShiftNetTeacher, load_pocketdvdnet_checkpoint
from workflow.artifacts import load_recipe
from workflow.config import dump_json
from workflow.distill import run as run_distill
from workflow.evaluate import run as run_evaluate
from workflow.prune import run as run_prune
from workflow.runtime import (
    added_noise_to_teacher_map,
    pack_frames,
    select_device,
    unpack_packed_frames,
)
from workflow.teacher import run as run_teacher


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_RECIPE_PATH = REPO_ROOT / "configs" / "paper" / "pocketdvdnet_recipe.json"
NOISE_CSV_PATH = REPO_ROOT / "dataloaders" / "predicted_labels.csv"
SHIFT_NET_DIR = REPO_ROOT / "external" / "Shift-Net"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny smoke checks for the staged paper workflow."
    )
    parser.add_argument(
        "--mode",
        choices=("all", "stages", "checkpoints"),
        default="all",
        help="Which smoke checks to run.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root. Falls back to the DATA_ROOT environment variable.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory containing downloaded paper checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/smoke",
        help="Directory for smoke-test outputs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device override. Defaults to auto.",
    )
    return parser.parse_args()


def resolve_data_root(cli_value: str | None) -> Path:
    data_root = cli_value or os.environ.get("DATA_ROOT")

    if not data_root:
        raise ValueError("DATA_ROOT is required. Set the environment variable or pass --data-root.")

    path = Path(data_root).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {path}")
    return path


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def first_sequence_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    child_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    sequence_dirs = [path for path in child_dirs if image_files(path)]
    if sequence_dirs:
        return sequence_dirs[0]

    if image_files(root):
        return root

    raise RuntimeError(f"No image sequences found in {root}")


def copy_sequence(source_dir: Path, target_dir: Path, max_frames: int) -> int:
    frames = image_files(source_dir)[:max_frames]
    if len(frames) < max_frames:
        raise RuntimeError(
            f"Expected at least {max_frames} frames in {source_dir}, found {len(frames)}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        shutil.copy2(frame, target_dir / frame.name)
    return len(frames)


def prepare_tiny_subset(
    data_root: Path, subset_root: Path, frames_per_sequence: int = 5
) -> dict[str, object]:
    if subset_root.exists():
        shutil.rmtree(subset_root)

    train_source = first_sequence_dir(data_root / "train")
    val_source = first_sequence_dir(data_root / "DAVIS_val")

    train_target = subset_root / "train" / "sequence_000"
    val_target = subset_root / "DAVIS_val" / "sequence_000"
    copied_train = copy_sequence(train_source, train_target, max_frames=frames_per_sequence)
    copied_val = copy_sequence(val_source, val_target, max_frames=frames_per_sequence)

    return {
        "trainset_dir": str((subset_root / "train").resolve()),
        "valset_dir": str((subset_root / "DAVIS_val").resolve()),
        "copied_train_frames": copied_train,
        "copied_val_frames": copied_val,
    }


def stage_smoke_configs(
    dataset_paths: dict[str, object],
    root: Path,
    device: str,
) -> tuple[dict, dict, dict, dict]:
    prune_root = root / "prune"
    teacher_root = root / "teacher"
    distill_root = root / "distill"
    eval_root = root / "eval"

    prune_config = {
        "seed": 1,
        "device": device,
        "sequence_length": 5,
        "trainset_dir": dataset_paths["trainset_dir"],
        "valset_dir": dataset_paths["valset_dir"],
        "output_dir": str(prune_root),
        "recipe_path": str(prune_root / "pocketdvdnet_recipe.json"),
        "batch_size": 1,
        "patch_size": 64,
        "iterations": 2,
        "num_workers": 0,
        "num_prefetch_queue": 1,
        "val_sequence_length": 5,
        "val_interval": 1,
        "scheduler_step_interval": 1,
        "lr": 1.5e-5,
        "noise_ival": [5, 55],
        "val_noiseL": 25.0,
        "optimizer": {
            "lambda_reg": 0.09,
            "epoch_size": 1,
            "np_steps": 2,
            "no_steps": "inf",
            "eps": 0.001,
            "weight_decay": 0.0,
            "lambda_warmup_steps": 1,
        },
        "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 0.8},
    }
    teacher_config = {
        "seed": 1,
        "device": device,
        "sequence_length": 5,
        "trainset_dir": dataset_paths["trainset_dir"],
        "valset_dir": dataset_paths["valset_dir"],
        "noise_csv": str(NOISE_CSV_PATH),
        "external_shift_net_dir": str(SHIFT_NET_DIR),
        "teacher_arch": "gshift_denoise1",
        "output_dir": str(teacher_root),
        "batch_size": 1,
        "patch_size": 64,
        "iterations": 2,
        "num_workers": 0,
        "num_prefetch_queue": 1,
        "val_sequence_length": 5,
        "val_interval": 1,
        "scheduler_step_interval": 1,
        "noise_seed": 1,
        "lr": 1e-4,
        "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 0.6},
    }
    distill_config = {
        "seed": 1,
        "device": device,
        "sequence_length": 5,
        "trainset_dir": dataset_paths["trainset_dir"],
        "valset_dir": dataset_paths["valset_dir"],
        "noise_csv": str(NOISE_CSV_PATH),
        "external_shift_net_dir": str(SHIFT_NET_DIR),
        "teacher_arch": "gshift_denoise1",
        "teacher_checkpoint": str(teacher_root / "best.pt"),
        "recipe_path": str(prune_root / "pocketdvdnet_recipe.json"),
        "output_dir": str(distill_root),
        "batch_size": 1,
        "patch_size": 64,
        "iterations": 2,
        "num_workers": 0,
        "num_prefetch_queue": 1,
        "val_sequence_length": 5,
        "val_interval": 1,
        "scheduler_step_interval": 1,
        "noise_seed": 1,
        "lr": 1e-3,
        "distill_alpha": 0.5,
        "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 0.6},
    }
    eval_config = {
        "device": device,
        "sequence_length": 5,
        "valset_dir": dataset_paths["valset_dir"],
        "noise_csv": str(NOISE_CSV_PATH),
        "recipe_path": str(prune_root / "pocketdvdnet_recipe.json"),
        "checkpoint_path": str(distill_root / "best.pt"),
        "output_dir": str(eval_root),
        "val_sequence_length": 5,
        "noise_seed": 1,
    }
    return prune_config, teacher_config, distill_config, eval_config


def checkpoint_smoke_configs(
    dataset_paths: dict[str, object],
    root: Path,
    checkpoint_dir: Path,
    device: str,
) -> tuple[dict, dict]:
    distill_config = {
        "seed": 1,
        "device": device,
        "sequence_length": 5,
        "trainset_dir": dataset_paths["trainset_dir"],
        "valset_dir": dataset_paths["valset_dir"],
        "noise_csv": str(NOISE_CSV_PATH),
        "external_shift_net_dir": str(SHIFT_NET_DIR),
        "teacher_arch": "gshift_denoise1",
        "teacher_checkpoint": str((checkpoint_dir / "shift-net_retrained.pth").resolve()),
        "recipe_path": str(PUBLISHED_RECIPE_PATH),
        "output_dir": str(root / "distill_from_downloaded_teacher"),
        "batch_size": 1,
        "patch_size": 64,
        "iterations": 2,
        "num_workers": 0,
        "num_prefetch_queue": 1,
        "val_sequence_length": 5,
        "val_interval": 1,
        "scheduler_step_interval": 1,
        "noise_seed": 1,
        "lr": 1e-3,
        "distill_alpha": 0.5,
        "scheduler": {"name": "StepLR", "step_size": 1, "gamma": 0.6},
    }
    eval_config = {
        "device": device,
        "sequence_length": 5,
        "valset_dir": dataset_paths["valset_dir"],
        "noise_csv": str(NOISE_CSV_PATH),
        "recipe_path": str(PUBLISHED_RECIPE_PATH),
        "checkpoint_path": str((checkpoint_dir / "pocketdvdnet.pt").resolve()),
        "output_dir": str(root / "eval_downloaded_student"),
        "val_sequence_length": 5,
        "noise_seed": 1,
    }
    return distill_config, eval_config


def load_first_validation_sequence(valset_dir: str) -> torch.Tensor:
    dataset = ValDataset(valsetdir=valset_dir, gray_mode=False, num_input_frames=5)
    if len(dataset) == 0:
        raise RuntimeError(f"No validation sequences found in {valset_dir}")
    return dataset[0].unsqueeze(0)


def smoke_teacher_checkpoint(
    checkpoint_path: Path, valset_dir: str, device: torch.device
) -> dict[str, object]:
    teacher = ShiftNetTeacher(SHIFT_NET_DIR, arch="gshift_denoise1").to(device)
    teacher.load_checkpoint(checkpoint_path, map_location=device)
    clean_sequence = load_first_validation_sequence(valset_dir).to(device)
    noise_model = NoiseModel(
        str(NOISE_CSV_PATH),
        seed=1,
        num_frames=clean_sequence.shape[1],
    )
    noisy_packed, added_noise = noise_model(pack_frames(clean_sequence))
    noisy_sequence = unpack_packed_frames(noisy_packed, num_frames=clean_sequence.shape[1])
    teacher_noise_map = added_noise_to_teacher_map(
        added_noise,
        num_frames=clean_sequence.shape[1],
    )
    with torch.no_grad():
        prediction = teacher.forward_center_frame(
            noisy_sequence.to(device),
            teacher_noise_map.to(device),
        )
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "loaded": True,
        "prediction_shape": list(prediction.shape),
    }


def smoke_student_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    model = PocketDVDNet.from_recipe(load_recipe(PUBLISHED_RECIPE_PATH)).to(device)
    load_pocketdvdnet_checkpoint(model, checkpoint_path, map_location=device)
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "loaded": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def ensure_checkpoint_files(checkpoint_dir: Path) -> None:
    required = [
        checkpoint_dir / "shift-net_retrained.pth",
        checkpoint_dir / "pocketdvdnet.pt",
    ]
    missing = [str(path.resolve()) for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing downloaded checkpoint files:\n{joined}")


def run_stage_smoke(data_root: Path, output_dir: Path, device: str) -> dict[str, object]:
    dataset_paths = prepare_tiny_subset(data_root, output_dir / "data")
    prune_config, teacher_config, distill_config, eval_config = stage_smoke_configs(
        dataset_paths,
        output_dir / "outputs",
        device,
    )

    run_prune(prune_config)
    run_teacher(teacher_config)
    run_distill(distill_config)
    run_evaluate(eval_config)

    return {
        "mode": "stages",
        "dataset": dataset_paths,
        "outputs": {
            "prune_recipe": prune_config["recipe_path"],
            "teacher_checkpoint": str(Path(teacher_config["output_dir"]) / "best.pt"),
            "distill_checkpoint": str(Path(distill_config["output_dir"]) / "best.pt"),
            "eval_metrics": str(Path(eval_config["output_dir"]) / "metrics.json"),
        },
    }


def run_checkpoint_smoke(
    data_root: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, object]:
    ensure_checkpoint_files(checkpoint_dir)
    dataset_paths = prepare_tiny_subset(data_root, output_dir / "data")
    distill_config, eval_config = checkpoint_smoke_configs(
        dataset_paths,
        output_dir / "outputs",
        checkpoint_dir,
        device_name,
    )
    device = select_device(device_name)

    teacher_summary = smoke_teacher_checkpoint(
        checkpoint_dir / "shift-net_retrained.pth",
        str(dataset_paths["valset_dir"]),
        device,
    )
    student_summary = smoke_student_checkpoint(checkpoint_dir / "pocketdvdnet.pt", device)
    run_distill(distill_config)
    run_evaluate(eval_config)

    return {
        "mode": "checkpoints",
        "dataset": dataset_paths,
        "teacher_checkpoint": teacher_summary,
        "student_checkpoint": student_summary,
        "outputs": {
            "distill_resume_checkpoint": str(Path(distill_config["output_dir"]) / "best.pt"),
            "eval_metrics": str(Path(eval_config["output_dir"]) / "metrics.json"),
        },
    }


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "device": str(select_device(args.device)),
        "output_dir": str(output_dir),
    }
    if args.mode in {"all", "stages"}:
        summary["stages"] = run_stage_smoke(data_root, output_dir / "stages", args.device)
    if args.mode in {"all", "checkpoints"}:
        summary["checkpoints"] = run_checkpoint_smoke(
            data_root, checkpoint_dir, output_dir / "checkpoints", args.device
        )

    dump_json(output_dir / "summary.json", summary)
    print(f"[verify_smoke] wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
