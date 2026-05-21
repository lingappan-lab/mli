"""Configuration, export-root, and example-image helpers for the Gradio UI."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from PIL import Image as PILImage

from lingappan_mli import DEFAULT_PIXEL_HEIGHT_UM, DEFAULT_PIXEL_WIDTH_UM
from lingappan_mli.io import SUPPORTED_EXTENSIONS

ROOT = Path(__file__).resolve().parents[3]

EXPORT_ROOT_ENV = "LINGAPPAN_MLI_EXPORT_ROOT"
ALLOW_UNSAFE_EXPORT_ROOT_ENV = "LINGAPPAN_MLI_ALLOW_UNSAFE_EXPORT_ROOT"
EXPORT_RUN_PREFIX = "lingappan_mli_"
DEFAULT_EXPORT_ROOT = Path(tempfile.gettempdir()) / "lingappan_mli_exports"


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative integer, got {value!r}.") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}.")
    return parsed


EXPORT_RUN_TTL_SECONDS = _positive_int_env("LINGAPPAN_MLI_EXPORT_TTL_SECONDS", 30 * 60)
GRADIO_CACHE_CLEANUP_SECONDS = _positive_int_env("LINGAPPAN_MLI_GRADIO_CACHE_CLEANUP_SECONDS", 10 * 60)
GRADIO_CACHE_TTL_SECONDS = _positive_int_env("LINGAPPAN_MLI_GRADIO_CACHE_TTL_SECONDS", 10 * 60)
MAX_UPLOAD_COUNT = _positive_int_env("LINGAPPAN_MLI_MAX_UPLOAD_COUNT", 200)
MAX_UPLOAD_FILE_BYTES = _positive_int_env("LINGAPPAN_MLI_MAX_UPLOAD_FILE_BYTES", 250 * 1024 * 1024)
MAX_UPLOAD_TOTAL_BYTES = _positive_int_env(
    "LINGAPPAN_MLI_MAX_UPLOAD_TOTAL_BYTES", 2 * 1024 * 1024 * 1024
)
MAX_UPLOAD_PIXELS = _positive_int_env("LINGAPPAN_MLI_MAX_UPLOAD_PIXELS", 100_000_000)
MAX_UPLOAD_DIMENSION = _positive_int_env("LINGAPPAN_MLI_MAX_UPLOAD_DIMENSION", 50_000)


def path_is_relative_to(path: Path, root: Path) -> bool:
    return path.is_relative_to(root)


def _unsafe_export_root_reasons(path: Path) -> list[str]:
    resolved = path.expanduser().resolve(strict=False)
    reasons: list[str] = []
    if resolved.parent == resolved:
        reasons.append("filesystem root")
    home = Path.home().resolve(strict=False)
    if resolved == home:
        reasons.append("home directory")
    tmp_root = Path(tempfile.gettempdir()).resolve(strict=False)
    if resolved == tmp_root:
        reasons.append("temporary directory root; use a nested app directory")
    app_root = ROOT.resolve(strict=False)
    if resolved == app_root or path_is_relative_to(app_root, resolved):
        reasons.append("application source tree or one of its parents")
    if path.is_symlink():
        reasons.append("symlink")
    return reasons


def resolve_export_root() -> Path:
    raw_root = os.environ.get(EXPORT_ROOT_ENV, str(DEFAULT_EXPORT_ROOT))
    requested = Path(raw_root).expanduser()
    resolved = requested.resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise RuntimeError(f"{EXPORT_ROOT_ENV} must point to a directory, not a file: {resolved}")
    reasons = _unsafe_export_root_reasons(requested)
    if reasons and os.environ.get(ALLOW_UNSAFE_EXPORT_ROOT_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}:
        reason_text = ", ".join(reasons)
        raise RuntimeError(
            f"Refusing unsafe export root {resolved} ({reason_text}). "
            f"Choose a dedicated export directory or set {ALLOW_UNSAFE_EXPORT_ROOT_ENV}=1 to override explicitly."
        )
    return resolved


def cleanup_stale_export_runs(
    export_root: Path,
    ttl_seconds: int = EXPORT_RUN_TTL_SECONDS,
    now: float | None = None,
) -> int:
    """Remove stale direct child export run directories created by this app."""
    if ttl_seconds <= 0:
        return 0
    root = Path(export_root).resolve(strict=False)
    if not root.exists():
        return 0
    cutoff = (time.time() if now is None else now) - ttl_seconds
    removed = 0
    for child in root.iterdir():
        if not child.name.startswith(EXPORT_RUN_PREFIX) or child.is_symlink() or not child.is_dir():
            continue
        try:
            mtime = child.stat(follow_symlinks=False).st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        try:
            shutil.rmtree(child)
            removed += 1
        except OSError:
            continue
    return removed


def remove_export_run(run_dir: str | Path | None, export_root: Path | None = None) -> bool:
    """Remove one app-created export run directory if it is safe to delete."""
    if not run_dir:
        return False
    root = Path(EXPORT_ROOT if export_root is None else export_root).resolve(strict=False)
    run_path = Path(run_dir).expanduser().resolve(strict=False)
    if run_path == root or not path_is_relative_to(run_path, root):
        return False
    if not run_path.name.startswith(EXPORT_RUN_PREFIX) or run_path.is_symlink() or not run_path.is_dir():
        return False
    try:
        shutil.rmtree(run_path)
    except OSError:
        return False
    return True


EXPORT_ROOT = resolve_export_root()
EXAMPLE_INPUT_ROOT = ROOT / "example_input"
EXAMPLE_PIXEL_WIDTH_UM = DEFAULT_PIXEL_WIDTH_UM
EXAMPLE_PIXEL_HEIGHT_UM = DEFAULT_PIXEL_HEIGHT_UM
EXAMPLE_LINE_SPACING_UM = 35.4


def _find_example_image() -> Path | None:
    if not EXAMPLE_INPUT_ROOT.exists():
        return None
    for path in sorted(EXAMPLE_INPUT_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return path
    return None


def _example_slide_field(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    if "_" in stem:
        slide_id, field_id = stem.rsplit("_", 1)
        return slide_id or stem, field_id or "field"
    return stem, "field"


def _load_example_preview(path: Path | None):
    if path is None:
        return None
    try:
        with PILImage.open(path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((760, 620))
            return preview.copy()
    except Exception:
        return None


def _image_dimensions(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        with PILImage.open(path) as image:
            return image.size
    except Exception:
        return None


EXAMPLE_IMAGE_PATH = _find_example_image()
EXAMPLE_IMAGE_PREVIEW = _load_example_preview(EXAMPLE_IMAGE_PATH)
EXAMPLE_IMAGE_DIMS = _image_dimensions(EXAMPLE_IMAGE_PATH)
EXAMPLE_FILENAME = EXAMPLE_IMAGE_PATH.name if EXAMPLE_IMAGE_PATH is not None else "example-A_0000.tif"
EXAMPLE_SLIDE_ID, EXAMPLE_FIELD_ID = _example_slide_field(EXAMPLE_FILENAME)
