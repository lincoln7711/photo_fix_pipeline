#!/usr/bin/env python3
"""
Photogrammetry Post-Processing Pipeline
────────────────────────────────────────
Stage 1  Background removal     (rembg / u2net)
Stage 2  Exposure normalization  (Pillow + OpenCV)
Stage 3  3-D reconstruction      (Agisoft Metashape Python API)
Stage 4  Texture enhancement     (Stable Diffusion img2img)

Usage:
    python main.py --input ./photos
    python main.py --input ./photos --stages 1 2
    python main.py --input ./photos --stages 3 4 --dry-run
    python main.py --input ./photos --config custom_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_images(folder: Path, extensions: Sequence[str]) -> list[Path]:
    """Glob a folder for files with any of the given extensions (case-insensitive)."""
    images: list[Path] = []
    for ext in extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


# ── Stage 1 — Background Removal ───────────────────────────────────────────────

def stage_rembg(
    input_dir: Path,
    output_root: Path,
    cfg: dict,
    dry_run: bool,
) -> Path:
    out = output_root / cfg["paths"]["stage1_dir"]
    images = collect_images(input_dir, [".jpg", ".jpeg"])

    if not images:
        raise FileNotFoundError(f"No JPG/JPEG images found in: {input_dir}")

    log.info("[Stage 1] Background removal — %d image(s) → %s", len(images), out)

    if dry_run:
        for img in images:
            log.info("  [dry-run] %s → %s", img.name, img.stem + ".png")
        return out

    out.mkdir(parents=True, exist_ok=True)

    # Deferred import so the script can be imported / dry-run without rembg installed
    from rembg import new_session, remove  # noqa: PLC0415

    model_name: str = cfg["rembg"].get("model", "u2net")
    log.info("  Loading rembg session: %s", model_name)
    session = new_session(model_name)

    for idx, img_path in enumerate(images, 1):
        out_path = out / (img_path.stem + ".png")
        log.info("  [%d/%d] %s → %s", idx, len(images), img_path.name, out_path.name)
        out_path.write_bytes(remove(img_path.read_bytes(), session=session))

    log.info("[Stage 1] Done — output: %s", out)
    return out


# ── Stage 2 — Exposure Normalization ───────────────────────────────────────────

def stage_normalize(
    input_dir: Path,
    output_root: Path,
    cfg: dict,
    dry_run: bool,
) -> Path:
    out = output_root / cfg["paths"]["stage2_dir"]

    # In a dry-run the preceding stage never wrote anything, so input_dir may
    # not exist yet.  Report the planned operation and move on.
    if dry_run and not input_dir.exists():
        log.info("[Stage 2] Exposure normalization — input: %s → %s", input_dir, out)
        log.info("  [dry-run] Would normalize all PNGs produced by Stage 1")
        return out

    images = collect_images(input_dir, [".png"])

    if not images:
        raise FileNotFoundError(f"No PNG images found in: {input_dir}")

    log.info("[Stage 2] Exposure normalization — %d image(s) → %s", len(images), out)

    if dry_run:
        for img in images:
            log.info("  [dry-run] %s", img.name)
        return out

    import cv2          # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    norm_cfg = cfg["normalization"]

    # ── Pass 1: compute dataset-wide target L* brightness ──────────────────────
    # We work in CIE LAB so brightness shifts are perceptually uniform.
    # Transparent pixels (alpha == 0) are excluded from the mean.
    bright_samples: list[float] = []

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3:
            continue
        bgr = img[:, :, :3]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        if img.shape[2] == 4:
            mask = img[:, :, 3] > 0
            if mask.any():
                bright_samples.append(float(l_channel[mask].mean()))
        else:
            bright_samples.append(float(l_channel.mean()))

    if not bright_samples:
        raise RuntimeError("Could not compute brightness — all images may be empty or unreadable.")

    dataset_mean = float(np.mean(bright_samples))
    target_l = norm_cfg.get("target_brightness") or dataset_mean
    log.info(
        "  Target L* brightness: %.1f  (dataset mean: %.1f, %d frames sampled)",
        target_l, dataset_mean, len(bright_samples),
    )

    clip_limit = float(norm_cfg.get("clahe_clip_limit", 2.0))
    tile_size  = tuple(int(x) for x in norm_cfg.get("clahe_tile_size", [8, 8]))

    # ── Pass 2: CLAHE + brightness shift ───────────────────────────────────────
    for idx, img_path in enumerate(images, 1):
        out_path = out / img_path.name
        log.info("  [%d/%d] %s", idx, len(images), img_path.name)

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            log.warning("  Cannot read %s — skipping.", img_path.name)
            continue

        has_alpha = img.ndim == 3 and img.shape[2] == 4
        alpha = img[:, :, 3].copy() if has_alpha else None
        bgr   = img[:, :, :3]

        lab       = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b   = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_eq  = clahe.apply(l)

        # Shift mean brightness to match target
        shift = int(target_l - (float(l_eq.mean()) or 1.0))
        l_eq  = np.clip(l_eq.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        bgr_out = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
        out_img = np.dstack([bgr_out, alpha]) if has_alpha else bgr_out
        cv2.imwrite(str(out_path), out_img)

    log.info("[Stage 2] Done — output: %s", out)
    return out


# ── Stage 3 — Metashape Reconstruction ─────────────────────────────────────────

def stage_metashape(
    input_dir: Path,
    output_root: Path,
    cfg: dict,
    dry_run: bool,
) -> Path:
    out = output_root / cfg["paths"]["stage3_dir"]

    if dry_run and not input_dir.exists():
        log.info("[Stage 3] Metashape reconstruction — input: %s → %s", input_dir, out)
        log.info("  [dry-run] Would reconstruct from all PNGs produced by Stage 2")
        log.info("  [dry-run] Project: %s.psx", cfg["metashape"]["project_name"])
        return out

    images = collect_images(input_dir, [".png"])

    if not images:
        raise FileNotFoundError(f"No PNG images found in: {input_dir}")

    log.info("[Stage 3] Metashape reconstruction — %d image(s) → %s", len(images), out)

    if dry_run:
        log.info("  [dry-run] %d photos → %s", len(images), out)
        log.info("  [dry-run] Project: %s.psx", cfg["metashape"]["project_name"])
        return out

    # Metashape is a manually installed .whl — fail clearly if absent
    try:
        import Metashape  # noqa: PLC0415
    except ImportError:
        log.error(
            "Metashape Python API not found.\n"
            "  Install with:  pip install /path/to/Metashape-2.x.x-cp311-cp311-win_amd64.whl\n"
            "  Download from: https://www.agisoft.com/downloads/installer/"
        )
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    ms_cfg = cfg["metashape"]

    accuracy_map: dict[str, object] = {
        "HighestAccuracy": Metashape.HighestAccuracy,
        "HighAccuracy":    Metashape.HighAccuracy,
        "MediumAccuracy":  Metashape.MediumAccuracy,
        "LowAccuracy":     Metashape.LowAccuracy,
        "LowestAccuracy":  Metashape.LowestAccuracy,
    }
    accuracy = accuracy_map.get(
        ms_cfg.get("match_accuracy", "HighAccuracy"),
        Metashape.HighAccuracy,
    )

    doc          = Metashape.Document()
    project_path = out / f"{ms_cfg['project_name']}.psx"
    doc.save(str(project_path))

    chunk = doc.addChunk()
    chunk.addPhotos([str(p) for p in images])
    log.info("  Added %d photos to chunk.", len(images))

    log.info("  matchPhotos(accuracy=%s)…", ms_cfg.get("match_accuracy", "HighAccuracy"))
    chunk.matchPhotos(
        accuracy=accuracy,
        generic_preselection=True,
        reference_preselection=False,
    )

    log.info("  alignCameras…")
    chunk.alignCameras()

    downscale = ms_cfg.get("depth_map_downscale", 2)
    log.info("  buildDepthMaps(downscale=%s)…", downscale)
    chunk.buildDepthMaps(
        downscale=downscale,
        filter_mode=Metashape.MildFiltering,
    )

    log.info("  buildModel…")
    chunk.buildModel(source_data=Metashape.DepthMapsData)

    tex_size = ms_cfg.get("texture_size", 4096)
    log.info("  buildTexture(size=%s)…", tex_size)
    chunk.buildTexture(
        texture_size=tex_size,
        count=ms_cfg.get("texture_count", 1),
    )

    doc.save()

    model_path = out / f"{ms_cfg['project_name']}.obj"
    log.info("  Exporting OBJ → %s", model_path.name)
    chunk.exportModel(
        str(model_path),
        format=Metashape.ModelFormatOBJ,
        texture_format=Metashape.ImageFormatPNG,
        save_texture=True,
        save_uv=True,
        save_normals=True,
    )

    log.info("[Stage 3] Done — model: %s", model_path)
    return out


# ── Stage 4 — Texture Enhancement (Stable Diffusion img2img) ───────────────────

def stage_enhance(
    input_dir: Path,
    output_root: Path,
    cfg: dict,
    dry_run: bool,
) -> Path:
    out = output_root / cfg["paths"]["stage4_dir"]

    if dry_run and not input_dir.exists():
        log.info("[Stage 4] Texture enhancement — input: %s → %s", input_dir, out)
        log.info("  [dry-run] Would enhance all texture PNGs exported by Stage 3")
        log.info(
            "  [dry-run] model=%s  strength=%.2f  steps=%d",
            cfg["stable_diffusion"]["model_id"],
            float(cfg["stable_diffusion"].get("denoising_strength", 0.30)),
            int(cfg["stable_diffusion"].get("num_inference_steps", 30)),
        )
        return out

    # Texture maps exported by Metashape are PNGs alongside the OBJ
    textures = collect_images(input_dir, [".png"])

    if not textures:
        raise FileNotFoundError(f"No PNG texture files found in: {input_dir}")

    log.info("[Stage 4] Texture enhancement — %d texture(s) → %s", len(textures), out)

    if dry_run:
        for tex in textures:
            log.info("  [dry-run] %s", tex.name)
        return out

    import torch  # noqa: PLC0415
    from diffusers import StableDiffusionImg2ImgPipeline  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    sd_cfg = cfg["stable_diffusion"]

    # ── Device selection ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
        log.info("  Device: CUDA — %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        dtype  = torch.float32
        log.warning(
            "  CUDA not available — running on CPU.  "
            "Stage 4 will be very slow; consider running on the Windows/CUDA machine."
        )

    out.mkdir(parents=True, exist_ok=True)

    model_id   = sd_cfg["model_id"]
    cache_dir  = Path(sd_cfg.get("cache_dir", "~/.cache/huggingface")).expanduser()
    local_only = bool(sd_cfg.get("local_files_only", False))

    log.info("  Loading pipeline: %s  (local_files_only=%s)", model_id, local_only)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=str(cache_dir),
        local_files_only=local_only,
    ).to(device)

    if device == "cuda":
        # Reduce peak VRAM usage on 8 GB cards without meaningful quality loss
        pipe.enable_attention_slicing()

    strength        = float(sd_cfg.get("denoising_strength", 0.30))
    prompt          = sd_cfg.get(
        "prompt",
        "photorealistic texture map, sharp surface detail, consistent lighting, "
        "clean geometry, no seams, no artifacts, high fidelity",
    )
    negative_prompt = sd_cfg.get(
        "negative_prompt",
        "blurry, noisy, artifacts, watermark, seams, distorted, overexposed, "
        "underexposed, color banding",
    )
    guidance_scale  = float(sd_cfg.get("guidance_scale", 7.5))
    steps           = int(sd_cfg.get("num_inference_steps", 30))
    proc_res        = int(sd_cfg.get("processing_resolution", 768))  # SD 2.1 native = 768

    log.info(
        "  strength=%.2f  steps=%d  guidance=%.1f  proc_res=%d",
        strength, steps, guidance_scale, proc_res,
    )

    for idx, tex_path in enumerate(textures, 1):
        out_path = out / tex_path.name
        log.info("  [%d/%d] %s", idx, len(textures), tex_path.name)

        original = Image.open(tex_path).convert("RGB")
        orig_w, orig_h = original.size

        # Scale to processing resolution; dimensions must be multiples of 8
        scale  = min(proc_res / orig_w, proc_res / orig_h)
        proc_w = (int(orig_w * scale) // 8) * 8
        proc_h = (int(orig_h * scale) // 8) * 8
        working = original.resize((proc_w, proc_h), Image.LANCZOS)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=working,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        ).images[0]

        # Restore original resolution so downstream tooling sees the expected dimensions
        if result.size != (orig_w, orig_h):
            result = result.resize((orig_w, orig_h), Image.LANCZOS)

        result.save(str(out_path), format="PNG")
        log.info("    saved %s  (%d×%d)", out_path.name, orig_w, orig_h)

    log.info("[Stage 4] Done — output: %s", out)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Photogrammetry post-processing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
stages:
  1  Background removal    (rembg)
  2  Exposure normalization (Pillow + OpenCV)
  3  3-D reconstruction    (Agisoft Metashape)
  4  Texture enhancement   (Stable Diffusion)

examples:
  python main.py --input ./photos
  python main.py --input ./photos --stages 1 2
  python main.py --input ./photos --stages 3 4 --dry-run
  python main.py --input ./photos --output /mnt/raid/project --config custom.yaml
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path, required=True,
        help="Folder of source JPGs (Stage 1 input).",
    )
    parser.add_argument(
        "--stages", "-s",
        nargs="+", type=int, metavar="N",
        choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
        help="Which stages to run (default: 1 2 3 4).",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path, default=Path("config.yaml"),
        help="Path to YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path, default=None,
        help="Override output root directory from config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned operations without executing.",
    )
    return parser.parse_args()


def resolve_stage_input(
    stage_num: int,
    raw_input: Path,
    stage_dirs: dict[int, Path],
    dry_run: bool = False,
) -> Path:
    """Return the appropriate input directory for a given stage.

    In a live run, walks backwards through preceding stages to find the most
    recently produced output directory that exists and is non-empty.
    In a dry-run, assumes each preceding stage would have produced its output
    (since nothing is actually written), so existence checks are skipped.
    Falls back to ``raw_input`` if no preceding output is found.
    """
    if stage_num == 1:
        return raw_input
    for prev in range(stage_num - 1, 0, -1):
        candidate = stage_dirs[prev]
        if dry_run or (candidate.exists() and any(candidate.iterdir())):
            return candidate
    log.warning(
        "  No output from a preceding stage found — "
        "falling back to raw input dir: %s", raw_input,
    )
    return raw_input


def main() -> None:
    args = parse_args()

    # ── Validate CLI inputs ────────────────────────────────────────────────────
    if not args.input.is_dir():
        log.error("Input directory not found: %s", args.input)
        sys.exit(1)
    if not args.config.exists():
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    cfg = load_config(args.config)

    output_root: Path = (args.output or Path(cfg["paths"]["output_root"])).resolve()

    stage_dirs: dict[int, Path] = {
        1: output_root / cfg["paths"]["stage1_dir"],
        2: output_root / cfg["paths"]["stage2_dir"],
        3: output_root / cfg["paths"]["stage3_dir"],
        4: output_root / cfg["paths"]["stage4_dir"],
    }

    stage_fns = {
        1: stage_rembg,
        2: stage_normalize,
        3: stage_metashape,
        4: stage_enhance,
    }

    stages = sorted(args.stages)

    if args.dry_run:
        log.info("=" * 60)
        log.info("DRY RUN — no files will be read or written")
        log.info("=" * 60)

    log.info("Input  : %s", args.input.resolve())
    log.info("Output : %s", output_root)
    log.info("Stages : %s", stages)
    log.info("Config : %s", args.config.resolve())

    t0 = time.perf_counter()

    for stage_num in stages:
        s_input = resolve_stage_input(stage_num, args.input.resolve(), stage_dirs, args.dry_run)
        log.info("─" * 60)
        stage_fns[stage_num](s_input, output_root, cfg, args.dry_run)

    elapsed = time.perf_counter() - t0
    log.info("─" * 60)
    log.info("Pipeline complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
