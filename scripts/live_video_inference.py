from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import PocketDVDNet, load_pocketdvdnet_checkpoint
from workflow.artifacts import load_recipe
from workflow.inference import tiled_forward_center_frame
from workflow.runtime import select_device


DEFAULT_RECIPE_PATH = REPO_ROOT / "configs" / "paper" / "pocketdvdnet_recipe.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "pocketdvdnet.pt"
TEMPORAL_WINDOW = 5


def parse_source(source: str) -> int | str:
    source = source.strip()
    if source.isdigit():
        return int(source)
    return source


def resolve_output_video_path(output_arg: str) -> Path:
    output_path = Path(output_arg).expanduser()
    if output_path.exists() and output_path.is_dir():
        return output_path / "live_denoised.mp4"
    if not output_path.suffix:
        return output_path / "live_denoised.mp4"
    return output_path


def resize_to_fit(frame: np.ndarray, max_width: int | None, max_height: int | None) -> np.ndarray:
    if max_width is None and max_height is None:
        return frame

    height, width = frame.shape[:2]
    scale = 1.0
    if max_width is not None:
        scale = min(scale, max_width / float(width))
    if max_height is not None:
        scale = min(scale, max_height / float(height))

    if scale >= 1.0:
        return frame

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def frame_to_tensor(
    frame_bgr: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).to(device=device)
    tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=dtype).div_(255.0)
    return tensor


def tensor_to_bgr_uint8(frame_rgb: torch.Tensor) -> np.ndarray:
    frame_uint8 = (
        frame_rgb.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)


def annotate_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    annotated = frame.copy()
    y = 28
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return annotated


def build_temporal_window(buffer: deque[torch.Tensor], window_size: int = TEMPORAL_WINDOW) -> list[torch.Tensor]:
    frames = list(buffer)
    if not frames:
        raise RuntimeError("Cannot build a temporal window from an empty buffer.")
    if len(frames) >= window_size:
        return frames[-window_size:]

    pad_left = (window_size - len(frames)) // 2
    pad_right = window_size - len(frames) - pad_left
    return [frames[0]] * pad_left + frames + [frames[-1]] * pad_right


class LatestFrameCapture:
    def __init__(self, source: int | str):
        self.source = source
        self.is_camera = isinstance(source, int) or str(source).startswith("/dev/video")
        if isinstance(source, str) and str(source).startswith("/dev/video"):
            self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        elif isinstance(source, int):
            self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        if self.is_camera:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stopped = False
        self._ended = False
        self._frame: np.ndarray | None = None
        self._timestamp = 0.0
        self._frame_index = -1
        self._has_unread_frame = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stopped:
            grabbed, frame = self.capture.read()
            if not grabbed or frame is None:
                if self.is_camera:
                    time.sleep(0.005)
                    continue
                with self._condition:
                    self._ended = True
                    self._condition.notify_all()
                break

            with self._condition:
                if not self.is_camera:
                    while self._has_unread_frame and not self._stopped:
                        self._condition.wait(timeout=0.01)
                    if self._stopped:
                        break
                self._frame = frame
                self._timestamp = time.perf_counter()
                self._frame_index += 1
                self._ended = False
                self._has_unread_frame = not self.is_camera
                self._condition.notify_all()

    def latest(self) -> tuple[np.ndarray | None, float, int, bool]:
        with self._lock:
            if self._frame is None:
                return None, 0.0, self._frame_index, self._ended
            return self._frame.copy(), self._timestamp, self._frame_index, self._ended

    def next_frame(self, timeout_seconds: float = 0.05) -> tuple[np.ndarray | None, float, int, bool]:
        if self.is_camera:
            return self.latest()

        deadline = time.perf_counter() + timeout_seconds
        with self._condition:
            while not self._has_unread_frame and not self._ended and not self._stopped:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            if not self._has_unread_frame:
                return None, 0.0, self._frame_index, self._ended

            frame = self._frame.copy()
            timestamp = self._timestamp
            frame_index = self._frame_index
            ended = self._ended
            self._has_unread_frame = False
            self._condition.notify_all()
            return frame, timestamp, frame_index, ended

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.capture.release()


def load_model_and_recipe(
    recipe_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    use_fp16: bool,
) -> torch.nn.Module:
    recipe = load_recipe(recipe_path)
    model = PocketDVDNet.from_recipe(recipe).to(device)
    load_pocketdvdnet_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    if use_fp16 and device.type == "cuda":
        model = model.half()
    return model


def maybe_compile_model(model: torch.nn.Module, enabled: bool, device: torch.device) -> torch.nn.Module:
    if not enabled or device.type != "cuda":
        return model

    if not hasattr(torch, "compile"):
        print("torch.compile is unavailable in this environment; continuing without it.")
        return model

    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception as exc:  # pragma: no cover - compile is an optional optimization
        print(f"torch.compile failed, continuing in eager mode: {exc}")
        return model


def infer_frame(
    model: torch.nn.Module,
    buffer: deque[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    window = build_temporal_window(buffer, TEMPORAL_WINDOW)
    packed = torch.stack(window, dim=0).reshape(1, TEMPORAL_WINDOW * 3, window[0].shape[1], window[0].shape[2])
    packed = packed.to(device=device, dtype=dtype)
    return tiled_forward_center_frame(model, packed, uses_noise_map=False).squeeze(0)


def run(args: argparse.Namespace) -> None:
    torch.backends.cudnn.benchmark = True

    device = select_device(args.device)
    fp16_enabled = args.fp16 if args.fp16 is not None else device.type == "cuda"
    dtype = torch.float16 if fp16_enabled and device.type == "cuda" else torch.float32

    model = load_model_and_recipe(args.recipe, args.checkpoint, device, fp16_enabled)
    model = maybe_compile_model(model, args.compile, device)

    source = parse_source(args.source)
    capture = LatestFrameCapture(source)
    buffer: deque[torch.Tensor] = deque(maxlen=TEMPORAL_WINDOW)
    writer: cv2.VideoWriter | None = None
    last_frame_index = -1
    last_overlay_update = time.perf_counter()
    frame_counter = 0
    processed_frame_counter = 0
    fps_value = 0.0
    display_name = "PocketDVDNet Live Demo"
    window_open = False

    try:
        while True:
            if capture.is_camera:
                frame_bgr, frame_ts, frame_index, ended = capture.latest()
                if frame_bgr is None:
                    if ended:
                        break
                    time.sleep(0.005)
                    continue

                if frame_index == last_frame_index:
                    if ended:
                        break
                    time.sleep(0.001)
                    continue
            else:
                frame_bgr, frame_ts, frame_index, ended = capture.next_frame()
                if frame_bgr is None:
                    if ended:
                        break
                    continue

            last_frame_index = frame_index
            frame_bgr = resize_to_fit(frame_bgr, args.max_width, args.max_height)
            frame_tensor = frame_to_tensor(frame_bgr, device, dtype)
            buffer.append(frame_tensor)

            start = time.perf_counter()
            with torch.inference_mode():
                output_rgb = infer_frame(model, buffer, device, dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_seconds = time.perf_counter() - start

            output_bgr = tensor_to_bgr_uint8(output_rgb)
            input_view = frame_bgr
            display_frame = output_bgr if args.denoised_only else np.hstack((input_view, output_bgr))

            overlay_seconds = time.perf_counter() - frame_ts
            frame_counter += 1
            processed_frame_counter += 1
            elapsed_overlay = time.perf_counter() - last_overlay_update
            if elapsed_overlay >= 1.0:
                fps_value = processed_frame_counter / elapsed_overlay
                processed_frame_counter = 0
                last_overlay_update = time.perf_counter()

            if args.overlay:
                buffer_status = f"buffer {min(len(buffer), TEMPORAL_WINDOW)}/{TEMPORAL_WINDOW}"
                fps_status = f"fps {fps_value:.1f}"
                latency_status = f"latency {overlay_seconds * 1000.0:.1f} ms"
                infer_status = f"infer {infer_seconds * 1000.0:.1f} ms"
                mode_status = "denoised only" if args.denoised_only else "input | denoised"
                display_frame = annotate_frame(
                    display_frame,
                    [
                        mode_status,
                        buffer_status,
                        fps_status,
                        latency_status,
                        infer_status,
                    ],
                )

            if writer is None and args.output_video is not None:
                writer = create_writer(args.output_video, display_frame.shape[1], display_frame.shape[0], capture)

            if writer is not None:
                writer.write(display_frame)

            if args.display:
                if not window_open:
                    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
                    window_open = True
                cv2.imshow(display_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if capture.is_camera and ended and frame_index == last_frame_index:
                break
    finally:
        capture.stop()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()


def create_writer(
    output_path: Path,
    width: int,
    height: int,
    capture: LatestFrameCapture,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    suffix = output_path.suffix.lower()
    if suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif suffix in {".mkv", ".webm"}:
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PocketDVDNet live/video demo inference")
    parser.add_argument("--source", default="0", help="Camera index, /dev/video path, or video file.")
    parser.add_argument(
        "--recipe",
        type=Path,
        default=DEFAULT_RECIPE_PATH,
        help="Path to the PocketDVDNet recipe JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the PocketDVDNet checkpoint.",
    )
    parser.add_argument("--device", default="auto", help="Torch device, or 'auto'.")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Force FP16 on CUDA.",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 even on CUDA.",
    )
    parser.set_defaults(fp16=None)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on CUDA if available.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional output video path. A directory will receive live_denoised.mp4.",
    )
    parser.add_argument(
        "--denoised-only",
        action="store_true",
        help="Show or write only the denoised stream instead of side-by-side input/denoised.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the preview window.",
    )
    parser.add_argument(
        "--overlay",
        dest="overlay",
        action="store_true",
        help="Overlay FPS and latency information on the output.",
    )
    parser.add_argument(
        "--no-overlay",
        dest="overlay",
        action="store_false",
        help="Disable FPS and latency overlay.",
    )
    parser.set_defaults(overlay=True)
    parser.add_argument("--max-width", type=int, default=None, help="Optional downscale width cap.")
    parser.add_argument("--max-height", type=int, default=None, help="Optional downscale height cap.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.output_video is not None:
        args.output_video = resolve_output_video_path(str(args.output_video))
    args.display = not args.no_display
    try:
        run(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
