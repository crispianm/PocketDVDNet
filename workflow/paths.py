from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIFT_NET_REPO_URL = "https://github.com/dasongli1/Shift-Net"
SHIFT_NET_COMMIT = "450a4f246dedccd306aa0bc02d615d797874e1ce"
DEFAULT_EXTERNAL_SHIFT_NET_DIR = REPO_ROOT / "external" / "Shift-Net"
LEGACY_SHIFT_NET_DIR = REPO_ROOT / "Shift-Net"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"
