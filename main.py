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

    # Metashape 2.x uses a downscale integer for matchPhotos instead of accuracy enums.
    # 0=Highest, 1=High, 2=Medium, 4=Low, 8=Lowest
    downscale_map: dict[str, int] = {
        "HighestAccuracy": 0,
        "HighAccuracy":    1,
        "MediumAccuracy":  2,
        "LowAccuracy":     4,
        "LowestAccuracy":  8,
    }
    match_downscale = downscale_map.get(
        ms_cfg.get("match_accuracy", "HighAccuracy"),
        1,
    )

    doc          = Metashape.Document()
    project_path = out / f"{ms_cfg['project_name']}.psx"
    doc.save(str(project_path))

    chunk = doc.addChunk()
    chunk.addPhotos([str(p) for p in images])
    log.info("  Added %d photos to chunk.", len(images))

    log.info("  matchPhotos(downscale=%s)…", match_downscale)
    chunk.matchPhotos(
        downscale=match_downscale,
        generic_preselection=True,
        reference_preselection=False,
    )

    log.info("  alignCameras…")
    chunk.alignCameras()
    doc.save()

    downscale = ms_cfg.get("depth_map_downscale", 2)
    log.info("  buildDepthMaps(downscale=%s)…", downscale)
    chunk.buildDepthMaps(
        downscale=downscale,
        filter_mode=Metashape.MildFiltering,
    )
    doc.save()

    log.info("  buildModel…")
    chunk.buildModel(source_data=Metashape.DepthMapsData)
    doc.save()

    min_component = int(ms_cfg.get("min_component_size", 1000))
    if min_component > 0:
        before = len(chunk.model.faces)
        chunk.model.removeComponents(min_component)
        after = len(chunk.model.faces)
        removed = before - after
        log.info(
            "  removeComponents(threshold=%d) — %d → %d faces (%d removed)",
            min_component, before, after, removed,
        )
        doc.save()

    tex_size = ms_cfg.get("texture_size", 4096)
    log.info("  buildTexture(size=%s)…", tex_size)
    chunk.buildTexture(
        texture_size=tex_size,
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


# ── Stage 4 — Texture Enhancement (Real-ESRGAN) ────────────────────────────────

# Model registry — maps config name → (arch kwargs, weights URL)
_REALESRGAN_MODELS: dict[str, tuple[dict, str]] = {
    "RealESRGAN_x4plus": (
        {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4},
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "RealESRGAN_x4plus_anime_6B": (
        {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 6, "num_grow_ch": 32, "scale": 4},
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
}


def stage_enhance(
    input_dir: Path,
    output_root: Path,
    cfg: dict,
    dry_run: bool,
) -> Path:
    out = output_root / cfg["paths"]["stage4_dir"]
    esr_cfg = cfg["realesrgan"]

    if dry_run and not input_dir.exists():
        log.info("[Stage 4] Texture enhancement — input: %s → %s", input_dir, out)
        log.info("  [dry-run] Would enhance all texture PNGs exported by Stage 3")
        log.info(
            "  [dry-run] model=%s  outscale=%s  tile=%s",
            esr_cfg.get("model_name", "RealESRGAN_x4plus"),
            esr_cfg.get("outscale", 1),
            esr_cfg.get("tile", 512),
        )
        return out

    textures = collect_images(input_dir, [".png"])

    if not textures:
        raise FileNotFoundError(f"No PNG texture files found in: {input_dir}")

    log.info("[Stage 4] Texture enhancement — %d texture(s) → %s", len(textures), out)

    if dry_run:
        for tex in textures:
            log.info("  [dry-run] %s", tex.name)
        return out

    import cv2                                          # noqa: PLC0415
    import numpy as np                                  # noqa: PLC0415
    import torch                                        # noqa: PLC0415
    from basicsr.archs.rrdbnet_arch import RRDBNet      # noqa: PLC0415
    from realesrgan import RealESRGANer                 # noqa: PLC0415

    model_name = esr_cfg.get("model_name", "RealESRGAN_x4plus")
    if model_name not in _REALESRGAN_MODELS:
        raise ValueError(f"Unknown realesrgan model_name: {model_name!r}. "
                         f"Choose from: {list(_REALESRGAN_MODELS)}")

    arch_kwargs, weights_url = _REALESRGAN_MODELS[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half   = bool(esr_cfg.get("half", True)) and device == "cuda"

    log.info("  Model: %s  device: %s  half: %s", model_name, device, half)

    rrdb_model = RRDBNet(**arch_kwargs)
    upsampler  = RealESRGANer(
        scale      = arch_kwargs["scale"],
        model_path = weights_url,
        model      = rrdb_model,
        tile       = int(esr_cfg.get("tile", 512)),
        tile_pad   = int(esr_cfg.get("tile_pad", 10)),
        pre_pad    = 0,
        half       = half,
        device     = device,
    )

    outscale = float(esr_cfg.get("outscale", 1))
    log.info("  outscale=%.0f  tile=%d  tile_pad=%d",
             outscale, int(esr_cfg.get("tile", 512)), int(esr_cfg.get("tile_pad", 10)))

    out.mkdir(parents=True, exist_ok=True)

    for idx, tex_path in enumerate(textures, 1):
        out_path = out / tex_path.name
        log.info("  [%d/%d] %s", idx, len(textures), tex_path.name)

        img = cv2.imread(str(tex_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            log.warning("  Cannot read %s — skipping.", tex_path.name)
            continue

        # Separate alpha if present; Real-ESRGAN operates on BGR only
        has_alpha = img.ndim == 3 and img.shape[2] == 4
        if has_alpha:
            alpha = img[:, :, 3]
            img   = img[:, :, :3]

        enhanced, _ = upsampler.enhance(img, outscale=outscale)

        if has_alpha:
            # Resize alpha to match enhanced dimensions and reattach
            ah, aw = enhanced.shape[:2]
            alpha_resized = cv2.resize(alpha, (aw, ah), interpolation=cv2.INTER_LINEAR)
            enhanced = np.dstack([enhanced, alpha_resized])

        cv2.imwrite(str(out_path), enhanced)
        h, w = enhanced.shape[:2]
        log.info("    saved %s  (%d×%d)", out_path.name, w, h)

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
