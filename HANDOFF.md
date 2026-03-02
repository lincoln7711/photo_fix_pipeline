# Handoff Briefing — photo_fix_pipeline

This document is written for a new Claude Code session picking up this project
on the Windows PC. Read this before touching anything else.

---

## What this project is

A four-stage Python 3.11 automation pipeline for photogrammetry post-processing:

| Stage | Task | Library |
|-------|------|---------|
| 1 | Background removal | `rembg` / u2net |
| 2 | Exposure normalization | `Pillow` + `OpenCV` (CLAHE + L* shift) |
| 3 | 3-D reconstruction | Agisoft Metashape Python API |
| 4 | Texture enhancement | `diffusers` / Stable Diffusion 2.1 img2img |

The script is cross-platform (pathlib throughout, no hardcoded separators).
It was authored on macOS M1 Pro and is intended to run on this Windows PC
with an NVIDIA RTX 3060 Ti (8 GB VRAM) and CUDA.

---

## Repo

**https://github.com/lincoln7711/photo_fix_pipeline**

Files in the repo:
```
main.py           pipeline orchestrator
config.yaml       all tunable parameters
requirements.txt  pinned dependencies (read the header before pip install)
README.md         full setup guide
HANDOFF.md        this file
.gitignore        excludes .venv/ and output/
```

---

## Current state

The pipeline is **complete and tested via dry-run on macOS**.

A dry-run against the real photo set completed successfully:

```
python main.py \
  --input "/path/to/nercomp-photogrammetry/Revised Shots/Chat MEasured" \
  --dry-run
```

- Stage 1 found and enumerated all 244 JPGs correctly
- Stages 2–4 chained correctly through their expected output directories
- No code changes are needed

The pipeline has **not yet been run live** on Windows. That is the next step.

---

## Windows environment — what we know

**Python version on this machine: 3.14**
**CUDA version on this machine: 12.9**

### Critical: Python 3.11 is required

The Metashape `.whl` is compiled for `cp311` (Python 3.11). It will hard-fail
on Python 3.14. Python 3.11 must be installed alongside 3.14.

**Install Python 3.11.9 from:**
https://www.python.org/downloads/release/python-3119/
Download the **Windows installer (64-bit)** and run it.

After install, verify with:
```powershell
py -3.11 --version    # must print Python 3.11.9
```

### Create the venv with 3.11 explicitly

```powershell
cd C:\path\to\photo_fix_pipeline
py -3.11 -m venv .venv
.venv\Scripts\activate       # if blocked, run first:
                              # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python --version              # must confirm 3.11.x
```

### PyTorch — use cu124 index (works on CUDA 12.9)

PyTorch does not publish cu129 wheels. CUDA drivers are forward-compatible,
so cu124 wheels run fine on a 12.9 system.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

No version pin — let it pull the latest stable for Python 3.11 + cu124.

Verify GPU is visible before proceeding:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 3060 Ti
```

### Install remaining dependencies

```powershell
pip install onnxruntime-gpu==1.19.2      # GPU-accelerated Stage 1
pip install -r requirements.txt           # everything else
```

`requirements.txt` lists `onnxruntime` (CPU); pip will warn about the conflict
with `onnxruntime-gpu` — that warning is safe to ignore.

### Metashape

Not on PyPI. Obtain the Python 3.11 wheel from Agisoft and install:
```powershell
pip install C:\path\to\Metashape-2.x.x-cp311-cp311-win_amd64.whl
```

### Pre-cache the Stable Diffusion model (~5 GB, needs internet once)

```powershell
python -c "from diffusers import StableDiffusionImg2ImgPipeline; StableDiffusionImg2ImgPipeline.from_pretrained('stabilityai/stable-diffusion-2-1')"
```

---

## Photo source

Photos are on a **network/server share**, not the local machine.

The directory structure is:
```
nercomp-photogrammetry/
├── Revised Shots/
│   ├── Chat MEasured/      ← 244 JPGs  (IMG_5447–IMG_5779)
│   ├── Chat Timed/         ← JPGs
│   ├── LM Measured/        ← JPGs
│   └── LM Timed/           ← JPGs
└── Initial Shots/
    ├── Chat MEasured/
    ├── Chat Timed/
    ├── LM MEasured/
    └── LM Timed/
```

**Each subject subfolder is a separate pipeline run.**
Point `--input` at the individual subject folder, not the parent.

The top level of `Revised Shots/` also contains a completed Metashape project
(`.psx`) and exported models (`.obj`, `.mtl`) from a previous manual run —
those are reference files, not pipeline inputs.

### Running per subject with separate outputs

```powershell
# Dry-run first to confirm paths resolve
python main.py --input "Z:\nercomp-photogrammetry\Revised Shots\Chat MEasured" --dry-run

# Live run
python main.py --input "Z:\nercomp-photogrammetry\Revised Shots\Chat MEasured" --output output\chat-measured
python main.py --input "Z:\nercomp-photogrammetry\Revised Shots\Chat Timed"    --output output\chat-timed
python main.py --input "Z:\nercomp-photogrammetry\Revised Shots\LM Measured"   --output output\lm-measured
python main.py --input "Z:\nercomp-photogrammetry\Revised Shots\LM Timed"      --output output\lm-timed
```

Replace `Z:\` with whatever drive letter or UNC path the server share maps to.

---

## Known issues / future improvements

- **Stage 4 texture tiling** — SD 2.1 processes at 768×768. Large texture maps
  (4096×4096) are currently downscaled for inference then upscaled back. At
  denoising strength ≤ 0.35 this is acceptable, but a tiled diffusion approach
  would give better results at full resolution. Not implemented yet.
- **Batch-all-subjects flag** — there is no `--recursive` flag to process all
  subject subfolders in one command. Each subject must be run separately.
  Worth adding if running all four subjects becomes repetitive.

---

## Requirements.txt note

After PyTorch installs successfully on this machine, run:
```powershell
pip show torch
```
and update `requirements.txt` with the exact version that installed, then
`git add requirements.txt && git commit && git push` so the repo reflects
the actual working version.
