from __future__ import annotations

from pathlib import Path


class MissingInputsError(RuntimeError):
    """Raised when a stage is missing required input files or directories."""


def _missing_entries(required_paths: dict[str, str | Path]) -> list[str]:
    missing: list[str] = []
    for label, raw_path in required_paths.items():
        path = Path(raw_path)
        if not path.exists():
            missing.append(f"{label}: {path}")
    return missing


def require_existing_paths(stage_label: str, required_paths: dict[str, str | Path]) -> None:
    missing = _missing_entries(required_paths)
    if not missing:
        return

    formatted = "\n".join(f"- {item}" for item in missing)
    raise MissingInputsError(f"{stage_label} is missing required inputs:\n{formatted}")


def collect_required_inputs(config: dict, stage: str) -> dict[str, str]:
    if stage == "prune":
        return {
            "training dataset": config["trainset_dir"],
            "validation dataset": config["valset_dir"],
        }

    if stage == "teacher":
        required = {
            "training dataset": config["trainset_dir"],
            "validation dataset": config["valset_dir"],
            "noise parameter csv": config["noise_csv"],
            "external Shift-Net checkout": config["external_shift_net_dir"],
        }
        pretrained_checkpoint = config.get("pretrained_checkpoint")
        if pretrained_checkpoint:
            required["teacher pretrained checkpoint"] = pretrained_checkpoint
        return required

    if stage == "distill":
        return {
            "training dataset": config["trainset_dir"],
            "validation dataset": config["valset_dir"],
            "noise parameter csv": config["noise_csv"],
            "external Shift-Net checkout": config["external_shift_net_dir"],
            "teacher checkpoint": config["teacher_checkpoint"],
            "PocketDVDNet recipe": config["recipe_path"],
        }

    if stage == "eval":
        required = {
            "validation dataset": config["valset_dir"],
            "PocketDVDNet recipe": config["recipe_path"],
            "distilled checkpoint": config["checkpoint_path"],
        }
        if config.get("noise_protocol", "csv_realistic") == "csv_realistic":
            required["noise parameter csv"] = config["noise_csv"]
        return required

    raise ValueError(f"Unknown stage: {stage}")


def validate_stage_inputs(config: dict, stage: str) -> None:
    require_existing_paths(f"Stage `{stage}`", collect_required_inputs(config, stage))
