# Photogrammetry Post-Processing Pipeline

Automates the full post-processing workflow for a photogrammetry capture session:

| Stage | Task | Key dependency |
|-------|------|----------------|
| 1 | Background removal | `rembg` / u2net |
| 2 | Exposure normalization | `Pillow` + `OpenCV` |
| 3 | 3-D reconstruction | Agisoft Metashape Python API |
| 4 | Texture enhancement | `diffusers` / Stable Diffusion 2.1 |

---

## Output structure

```
output/
├── 01_rembg/          PNGs with transparent backgrounds
├── 02_normalized/     Exposure-normalized PNGs
├── 03_model/          Metashape .psx project, .obj, .mtl, texture PNGs
└── 04_enhanced/       SD-enhanced texture maps
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11.x | Must be exactly 3.11 for Metashape .whl compatibility |
| CUDA Toolkit | 12.1 | Windows runtime machine only |
| Agisoft Metashape | 2.x | Requires a valid licence |
| Stable Diffusion 2.1 | — | Pre-cached locally; see below |

---

## macOS Setup (authoring / dry-run)

These steps prepare the environment for script development and dry-run validation.
Heavy processing (Stage 3, Stage 4) should be executed on the Windows machine.

### 1. Install pyenv

```bash
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zprofile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"'  >> ~/.zprofile
echo 'eval "$(pyenv init -)"' >> ~/.zprofile
source ~/.zprofile
```

### 2. Install Python 3.11

```bash
pyenv install 3.11.9
pyenv local 3.11.9        # pins version for this project directory
python --version           # should print Python 3.11.9
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 4. Install PyTorch (CPU — macOS)

```bash
pip install torch==2.4.1 torchvision==0.19.1
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify with a dry-run

```bash
python main.py --input ./sample_photos --dry-run
```

---

## Windows / CUDA Setup (runtime)

### 1. Install pyenv-win

```powershell
# Run in PowerShell (may require: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser)
Invoke-WebRequest -UseBasicParsing \
  "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" \
  -OutFile "./install-pyenv-win.ps1"
.\install-pyenv-win.ps1
```

Close and reopen PowerShell, then:

```powershell
pyenv install 3.11.9
pyenv local 3.11.9
python --version     # should print Python 3.11.9
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```

### 3. Install PyTorch with CUDA 12.1 support

**This step must be done before `requirements.txt` is installed.**

```powershell
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 `
  --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 3060 Ti
```

### 4. Install onnxruntime-gpu (faster Stage 1 on CUDA)

```powershell
pip install onnxruntime-gpu==1.19.2
```

> If you skip this step, rembg will still work using CPU onnxruntime — just slower.

### 5. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

> `requirements.txt` lists `onnxruntime` (CPU); if you installed `onnxruntime-gpu`
> above, pip will warn about a conflict — that warning is safe to ignore.

### 6. Install Agisoft Metashape Python API

Metashape is not published on PyPI. You must obtain the `.whl` from Agisoft.

1. Log in at [https://www.agisoft.com/downloads/installer/](https://www.agisoft.com/downloads/installer/)
2. Download the Python 3.11 wheel for your Metashape version, e.g.:
   `Metashape-2.1.3-cp311-cp311-win_amd64.whl`
3. Install it:

```powershell
pip install C:\Downloads\Metashape-2.1.3-cp311-cp311-win_amd64.whl
```

4. Verify:

```powershell
python -c "import Metashape; print(Metashape.app.version)"
```

> A valid Metashape licence must be activated on the machine.
> See [https://www.agisoft.com/support/activation/](https://www.agisoft.com/support/activation/)

### 7. Pre-cache the Stable Diffusion model

Stage 4 expects the model to be downloaded locally before the pipeline runs
(especially important for air-gapped machines). Run this once with internet access:

```powershell
python - <<'EOF'
from diffusers import StableDiffusionImg2ImgPipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1"
)
print("Model cached successfully.")
EOF
```

The model is stored in `~/.cache/huggingface` (~5 GB).
Once cached, set `local_files_only: true` in `config.yaml` to prevent any
network calls during pipeline execution.

---

## Usage

### Run the full pipeline

```bash
python main.py --input ./photos
```

### Run specific stages only

```bash
# Stages 1 and 2 only
python main.py --input ./photos --stages 1 2

# Stage 4 only (expects Stage 3 output to already exist in output/03_model/)
python main.py --input ./photos --stages 4
```

### Dry-run (validate without executing)

```bash
python main.py --input ./photos --dry-run
python main.py --input ./photos --stages 3 4 --dry-run
```

### Custom output directory

```bash
python main.py --input ./photos --output D:\projects\scan_01
```

### Custom config file

```bash
python main.py --input ./photos --config high_quality.yaml
```

### Full options

```
python main.py --help

arguments:
  --input,  -i   Folder of source JPGs (Stage 1 input)            [required]
  --stages, -s   Stages to run: 1 2 3 4 (default: all four)
  --config, -c   Path to YAML config (default: config.yaml)
  --output, -o   Override output root directory from config
  --dry-run      Print planned operations without executing
```

---

## Configuration reference

All pipeline parameters live in `config.yaml`.

| Key | Default | Description |
|-----|---------|-------------|
| `paths.output_root` | `output` | Root output directory |
| `rembg.model` | `u2net` | rembg model (`u2net`, `u2netp`, `isnet-general-use`, …) |
| `normalization.target_brightness` | `null` | CIE L* target (null = dataset mean) |
| `normalization.clahe_clip_limit` | `2.0` | CLAHE contrast limit |
| `normalization.clahe_tile_size` | `[8,8]` | CLAHE tile grid |
| `metashape.match_accuracy` | `HighAccuracy` | Photo matching accuracy |
| `metashape.depth_map_downscale` | `2` | Depth map downscale factor (1=Highest, 2=High) |
| `metashape.texture_size` | `4096` | Texture atlas resolution |
| `metashape.texture_count` | `1` | Number of texture atlases |
| `stable_diffusion.model_id` | `stabilityai/stable-diffusion-2-1` | HuggingFace model ID |
| `stable_diffusion.denoising_strength` | `0.30` | img2img strength (0.25–0.35 recommended) |
| `stable_diffusion.num_inference_steps` | `30` | Diffusion steps |
| `stable_diffusion.guidance_scale` | `7.5` | CFG scale |
| `stable_diffusion.processing_resolution` | `768` | Working resolution for SD (SD 2.1 native = 768) |
| `stable_diffusion.local_files_only` | `false` | Block all network calls |

---

## Notes

**Stage 3 — Metashape licence**
The script calls `doc.save()` before any processing so a partial project is
preserved if the job is interrupted. Re-running Stage 3 will overwrite the
existing `.psx` file.

**Stage 4 — VRAM budget**
`pipe.enable_attention_slicing()` is enabled automatically on CUDA to keep peak
VRAM under 8 GB for the RTX 3060 Ti. If you hit OOM errors, reduce
`processing_resolution` to `512` in `config.yaml`.

**Stage 4 — texture resolution**
SD 2.1 processes at 768×768. The texture is downscaled for inference then
upscaled back to its original resolution. At `denoising_strength` ≤ 0.35 this
is imperceptible. For higher-quality enhancement at full resolution, a tiled
approach (tile diffusion) should be used — this is a known future improvement.

**Running stages out of order**
If you skip a stage, the pipeline automatically locates the most recent
preceding stage output directory. If none exists, it falls back to `--input`.
Always confirm the resolved input with `--dry-run` before a long run.
