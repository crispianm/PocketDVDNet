from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow.config import load_config
from workflow.paths import (
    DEFAULT_EXTERNAL_SHIFT_NET_DIR,
    LEGACY_SHIFT_NET_DIR,
    SHIFT_NET_COMMIT,
    SHIFT_NET_REPO_URL,
)
from workflow.validate import MissingInputsError, collect_required_inputs


def run_command(*command: str, cwd: str | Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def ensure_shift_net_clone(target_dir: Path) -> None:
    if target_dir.exists():
        run_command(
            "git", "-C", str(target_dir), "fetch", "--depth", "1", "origin", SHIFT_NET_COMMIT
        )
    else:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        run_command("git", "clone", SHIFT_NET_REPO_URL, str(target_dir))
    run_command("git", "-C", str(target_dir), "checkout", SHIFT_NET_COMMIT)


def validate_configs(config_paths: list[Path]) -> None:
    for config_path in config_paths:
        try:
            config = load_config(config_path)
        except Exception as exc:
            print(f"[bootstrap] warning: {config_path} is not fully resolvable yet: {exc}")
            continue

        try:
            required_inputs = collect_required_inputs(config, stage=config_path.stem)
            missing = [item for item in required_inputs.items() if not Path(item[1]).exists()]
            if missing:
                missing_lines = ", ".join(f"{label}={path}" for label, path in missing)
                print(f"[bootstrap] warning: {config_path.name} is missing inputs: {missing_lines}")
        except (MissingInputsError, ValueError) as exc:
            print(f"[bootstrap] warning: could not validate {config_path.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the uv environment and pinned Shift-Net dependency."
    )
    parser.add_argument("--skip-sync", action="store_true", help="Skip `uv sync`.")
    parser.add_argument(
        "--skip-shiftnet-install", action="store_true", help="Skip installing external Shift-Net."
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if not args.skip_sync:
        run_command("uv", "sync", cwd=repo_root)

    if LEGACY_SHIFT_NET_DIR.exists():
        print(
            f"[bootstrap] legacy local Shift-Net detected at {LEGACY_SHIFT_NET_DIR}; leaving it untouched."
        )

    ensure_shift_net_clone(DEFAULT_EXTERNAL_SHIFT_NET_DIR)

    if not args.skip_shiftnet_install:
        run_command(
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "-e",
            str(DEFAULT_EXTERNAL_SHIFT_NET_DIR),
            cwd=repo_root,
        )

    validate_configs(sorted((repo_root / "configs" / "paper").glob("*.yaml")))
    print(f"[bootstrap] ready: Shift-Net pinned at {SHIFT_NET_COMMIT}")


if __name__ == "__main__":
    main()
