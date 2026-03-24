# PocketDVDNet Paper Reproduction

This repository is a paper-only reproduction workflow for [PocketDVDNet: Realtime Video Denoising for Real Camera Noise](https://arxiv.org/html/2601.16780v1).

The repo mirrors the paper stages directly:

1. Prune `FastDVDNet` with `OBProxSG`
2. Retrain a `Shift-Net` teacher on the realistic multi-component noise model
3. Distill the final noise-map-free `PocketDVDNet`
4. Evaluate the distilled checkpoint

## Setup

This repo uses `uv` only.

```bash
uv run python scripts/bootstrap.py
```

What bootstrap does:

- runs `uv sync`
- clones a pinned `Shift-Net` commit into `external/Shift-Net`
- installs that external dependency into the same environment
- checks the paper configs for unresolved paths
- warns early if required datasets or prior-stage artifacts are missing

Set `DATA_ROOT` before running the stages:

```bash
export DATA_ROOT=/absolute/path/to/data
```

Optional downloaded paper checkpoints:

```text
./checkpoints/shift-net_retrained.pth
./checkpoints/pocketdvdnet.pt
```

Google Drive folder:

- `https://drive.google.com/drive/folders/1Wdt6XGlgTEQQxAOoicfnk463ZP__ox1I?usp=drive_link`

These checkpoints are optional convenience artifacts:

- `shift-net_retrained.pth` lets you skip stage 2 and go straight to stage 3.
- `pocketdvdnet.pt` lets you skip stage 3 and run evaluation directly.
- the paper-first path still retrains the Shift-Net teacher from scratch in stage 2.

Expected dataset layout:

```text
$DATA_ROOT/
├── train/
│   ├── sequence_000/
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   └── ...
│   └── ...
├── DAVIS_val/
│   ├── aerial/
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   └── ...
│   └── ...
└── Set8/
    ├── snowboard/
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── ...
    └── ...
```

Validation rules:

- training data must be clean frame sequences only
- validation sequences must contain image files directly inside each sequence directory
- all configs use repo-relative output paths and `${DATA_ROOT}` for datasets
- every stage validates its required inputs before training starts

## Evaluation Presets

Paper-default evaluation commands:

```bash
uv run python stages/04_evaluate.py --config configs/paper/eval.yaml
uv run python stages/04_evaluate.py --config configs/paper/eval_set8.yaml
uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student.yaml
uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student_set8.yaml
```

These configs use the legacy paper-style seed sweep (`1..99`) by default. Use `--fast` to run the same paper protocol with the representative seed `99` only:

```bash
uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student_set8.yaml --fast
```

Realistic CSV-driven evaluation remains available explicitly:

```bash
uv run python stages/04_evaluate.py --config configs/paper/eval_realistic.yaml
uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student_realistic.yaml
uv run python stages/04_evaluate.py --config configs/paper/eval_set8_realistic.yaml
uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student_set8_realistic.yaml
```

## Live Demo

Run the live/video demo with the published recipe and checkpoint:

```bash
uv run python scripts/live_video_inference.py --source 0
uv run python scripts/live_video_inference.py --source /path/to/video.mp4 --output-video artifacts/live/demo.mp4
```

The demo uses the current repo loader, a rolling 5-frame window, and tiled center-frame inference. CUDA is used automatically when available, with FP16 enabled by default on GPU for the fastest high-quality path.

## Stages

| Stage | Paper section | Command | Input artifact | Output artifact |
| --- | --- | --- | --- | --- |
| 1 | 2.1 Model Pruning | `uv run python stages/01_prune_fastdvdnet.py --config configs/paper/prune.yaml` | clean train set + DAVIS validation set | `artifacts/prune/best.pt`, `artifacts/prune/pocketdvdnet_recipe.json`, `artifacts/prune/summary.json` |
| 2 | 2.3 Multi-Component Noise + teacher preparation for 2.4 | `uv run python stages/02_train_teacher.py --config configs/paper/teacher.yaml` | clean train/val data + external `Shift-Net` | `artifacts/teacher/best.pt`, `artifacts/teacher/summary.json` |
| 3 | 2.4 Knowledge Distillation | `uv run python stages/03_distill_pocketdvdnet.py --config configs/paper/distill.yaml` | stage 1 recipe + stage 2 teacher checkpoint | `artifacts/distill/best.pt`, `artifacts/distill/summary.json` |
| 4 | 3 Results and Discussion | `uv run python stages/04_evaluate.py --config configs/paper/eval.yaml` | stage 1 recipe + stage 3 distilled checkpoint + DAVIS validation set | `artifacts/eval/metrics.json` |

Optional downloaded-checkpoint commands:

- `uv run python stages/03_distill_pocketdvdnet.py --config configs/paper/distill_from_downloaded_teacher.yaml`
- `uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student.yaml`
- `uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student.yaml --fast`
- `uv run python stages/04_evaluate.py --config configs/paper/eval_downloaded_student_realistic.yaml`
- `uv run python scripts/live_video_inference.py --source 0`

Optional maintainer smoke checks:

- `uv run python scripts/verify_smoke.py --mode stages`
- `uv run python scripts/verify_smoke.py --mode checkpoints --checkpoint-dir checkpoints`

## Exact Outputs

Stage 1 writes:

- `artifacts/prune/latest.pt`
- `artifacts/prune/best.pt`
- `artifacts/prune/pocketdvdnet_recipe.json`
- `artifacts/prune/summary.json`

Stage 2 writes:

- `artifacts/teacher/latest.pt`
- `artifacts/teacher/best.pt`
- `artifacts/teacher/summary.json`

Stage 3 writes:

- `artifacts/distill/latest.pt`
- `artifacts/distill/best.pt`
- `artifacts/distill/summary.json`

Stage 4 writes:

- `artifacts/eval/metrics.json`

## Notes

- `PocketDVDNet.forward(x)` is the final student interface. The student does not accept an explicit noise map.
- `FastDVDNet` is kept only as the pruning source model.
- `Shift-Net` is external and pinned through `scripts/bootstrap.py`; the old submodule flow is not used.
- stage 2 now defaults to retraining the Shift-Net teacher from scratch.
- `configs/paper/pocketdvdnet_recipe.json` is the committed published student recipe for optional checkpoint evaluation and resume paths.
- The recipe artifact produced in stage 1 is the contract between pruning and distillation.
