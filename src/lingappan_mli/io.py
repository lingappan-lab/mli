"""Image and filesystem utilities for Lingappan MLI."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIFF_X_RESOLUTION_TAG = 282
TIFF_Y_RESOLUTION_TAG = 283
TIFF_RESOLUTION_UNIT_TAG = 296
EXIF_ORIENTATIONS_THAT_TRANSPOSE_AXES = {5, 6, 7, 8}


@dataclass(frozen=True)
class ImageLoadMetadata:
    """Audit metadata captured while loading an image for analysis."""

    original_format: str
    original_mode: str
    original_dtype: str
    frame_index: int
    page_count: int | None
    scaling_applied: bool
    scaling_min: float | None
    scaling_max: float | None
    embedded_resolution_x: float | None = None
    embedded_resolution_y: float | None = None
    embedded_resolution_unit: str = ""
    embedded_pixel_width_um: float | None = None
    embedded_pixel_height_um: float | None = None
    embedded_resolution_source: str = ""
    output_dtype: str = "uint8"
    warnings: tuple[str, ...] = ()

    def as_summary_dict(self) -> dict[str, object]:
        """Return field-summary/log columns for image-load audit metadata."""
        return {
            "image_original_format": self.original_format,
            "image_original_mode": self.original_mode,
            "image_original_dtype": self.original_dtype,
            "image_output_dtype": self.output_dtype,
            "image_frame_index": int(self.frame_index),
            "image_page_count": self.page_count,
            "image_scaling_applied": bool(self.scaling_applied),
            "image_scaling_min": self.scaling_min,
            "image_scaling_max": self.scaling_max,
            "image_embedded_resolution_x": self.embedded_resolution_x,
            "image_embedded_resolution_y": self.embedded_resolution_y,
            "image_embedded_resolution_unit": self.embedded_resolution_unit,
            "image_embedded_pixel_width_um": self.embedded_pixel_width_um,
            "image_embedded_pixel_height_um": self.embedded_pixel_height_um,
            "image_embedded_resolution_source": self.embedded_resolution_source,
            "image_load_warnings": " | ".join(self.warnings),
        }


@dataclass(frozen=True)
class ImageLoadResult:
    """Loaded RGB image plus audit metadata."""

    rgb: np.ndarray
    metadata: ImageLoadMetadata


def safe_stem(path: str | Path) -> str:
    """Return a filesystem-safe stem for output folders/files."""
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "image"


def discover_images(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted list of supported image paths."""
    images: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            images.extend(
                child
                for child in p.rglob("*")
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(p)
    return sorted(dict.fromkeys(images))


def _format_intensity(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.6g}"


def _safe_page_count(img: Image.Image) -> int | None:
    try:
        page_count = getattr(img, "n_frames")
    except Exception:
        return None
    try:
        return int(page_count)
    except (TypeError, ValueError):
        return None


def _safe_array_dtype(img: Image.Image) -> str:
    try:
        return str(np.asarray(img).dtype)
    except Exception:
        return "unavailable"


def _safe_tiff_tag_value(img: Image.Image, tag: int) -> object | None:
    for tag_container_name in ("tag_v2", "tag"):
        tag_container = getattr(img, tag_container_name, None)
        if tag_container is None:
            continue
        try:
            return tag_container.get(tag)
        except Exception:
            continue
    return None


def _resolution_value_to_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            numerator = _resolution_value_to_float(value[0])
            denominator = _resolution_value_to_float(value[1])
            if numerator is not None and denominator not in {None, 0}:
                value = numerator / denominator
            else:
                return None
        elif len(value) == 1:
            return _resolution_value_to_float(value[0])
        else:
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _resolution_unit_label(value: object | None) -> str:
    if value is None:
        return "unknown"
    try:
        unit_int = int(value)
    except (TypeError, ValueError):
        unit_text = str(value).strip().lower()
        if unit_text in {"2", "inch", "in", "inches"}:
            return "inch"
        if unit_text in {"3", "cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
            return "centimeter"
        if unit_text in {"1", "none", "no absolute unit", "unitless"}:
            return "none"
        return unit_text or "unknown"
    return {1: "none", 2: "inch", 3: "centimeter"}.get(unit_int, "unknown")


def _resolution_unit_um(unit: str) -> float | None:
    if unit == "inch":
        return 25_400.0
    if unit == "centimeter":
        return 10_000.0
    return None


def _pixel_size_from_resolution(resolution: float | None, unit: str) -> float | None:
    unit_um = _resolution_unit_um(unit)
    if resolution is None or unit_um is None:
        return None
    return float(unit_um / resolution)


def _embedded_resolution_metadata(img: Image.Image, exif_orientation: int) -> dict[str, object]:
    x_resolution = _resolution_value_to_float(_safe_tiff_tag_value(img, TIFF_X_RESOLUTION_TAG))
    y_resolution = _resolution_value_to_float(_safe_tiff_tag_value(img, TIFF_Y_RESOLUTION_TAG))
    resolution_unit = _resolution_unit_label(_safe_tiff_tag_value(img, TIFF_RESOLUTION_UNIT_TAG))
    source = "TIFF resolution tags" if x_resolution is not None or y_resolution is not None else ""

    if not source:
        dpi = img.info.get("dpi")
        if isinstance(dpi, (list, tuple)) and len(dpi) >= 2:
            x_resolution = _resolution_value_to_float(dpi[0])
            y_resolution = _resolution_value_to_float(dpi[1])
        else:
            x_resolution = y_resolution = _resolution_value_to_float(dpi)
        if x_resolution is not None or y_resolution is not None:
            resolution_unit = "inch"
            source = "Pillow dpi metadata"

    if x_resolution is not None and y_resolution is None:
        y_resolution = x_resolution
    elif y_resolution is not None and x_resolution is None:
        x_resolution = y_resolution

    if int(exif_orientation) in EXIF_ORIENTATIONS_THAT_TRANSPOSE_AXES:
        x_resolution, y_resolution = y_resolution, x_resolution

    return {
        "embedded_resolution_x": x_resolution,
        "embedded_resolution_y": y_resolution,
        "embedded_resolution_unit": resolution_unit if source else "",
        "embedded_pixel_width_um": _pixel_size_from_resolution(x_resolution, resolution_unit),
        "embedded_pixel_height_um": _pixel_size_from_resolution(y_resolution, resolution_unit),
        "embedded_resolution_source": source,
    }


def _is_tiff(path: str | Path, image_format: str) -> bool:
    return image_format.upper() == "TIFF" or Path(path).suffix.lower() in {".tif", ".tiff"}


def _safe_exif_orientation(img: Image.Image) -> int:
    try:
        orientation = img.getexif().get(274, 1)
    except Exception:
        return 1
    try:
        return int(orientation)
    except (TypeError, ValueError):
        return 1


def _apply_exif_orientation(img: Image.Image, orientation: int) -> Image.Image:
    """Apply EXIF orientation to an in-memory frame copy."""
    operations = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    operation = operations.get(int(orientation))
    return img.transpose(operation) if operation is not None else img


def _scale_to_uint8_with_metadata(arr: np.ndarray) -> tuple[np.ndarray, bool, float | None, float | None]:
    """Scale arbitrary numeric image data to uint8 and return scaling metadata."""
    if arr.dtype == np.uint8:
        return arr, False, None, None
    arr_float = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr_float)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8), True, None, None
    lo = float(arr_float[finite].min())
    hi = float(arr_float[finite].max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8), True, lo, hi
    scaled = (arr_float - lo) * (255.0 / (hi - lo))
    scaled = np.where(finite, scaled, 0.0)
    return np.clip(scaled, 0, 255).astype(np.uint8), True, lo, hi


def _non_uint8_scaling_warning(
    *,
    original_mode: str,
    original_dtype: str,
    scaling_min: float | None,
    scaling_max: float | None,
) -> str:
    if scaling_min is None or scaling_max is None:
        return (
            f"Non-uint8 image data (mode {original_mode}, dtype {original_dtype}) had no finite "
            "intensity range and was converted to an all-zero uint8 image before thresholding."
        )
    if scaling_max <= scaling_min:
        return (
            f"Non-uint8 image data (mode {original_mode}, dtype {original_dtype}) had constant finite "
            f"intensity {_format_intensity(scaling_min)} and was converted to an all-zero uint8 image "
            "before thresholding."
        )
    return (
        f"Non-uint8 image data (mode {original_mode}, dtype {original_dtype}) was min-max scaled to "
        f"uint8 using finite intensity range [{_format_intensity(scaling_min)}, "
        f"{_format_intensity(scaling_max)}] before thresholding."
    )


def _finalize_rgb_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("Loaded image could not be converted to RGB")
    return arr[:, :, :3].astype(np.uint8, copy=False)


def _load_numeric_array_as_rgb(
    arr: np.ndarray,
    *,
    original_mode: str,
    original_dtype: str,
    warnings: list[str],
) -> tuple[np.ndarray, bool, float | None, float | None]:
    scaled, scaling_applied, scaling_min, scaling_max = _scale_to_uint8_with_metadata(arr)
    if scaling_applied:
        warnings.append(
            _non_uint8_scaling_warning(
                original_mode=original_mode,
                original_dtype=original_dtype,
                scaling_min=scaling_min,
                scaling_max=scaling_max,
            )
        )
    return _finalize_rgb_array(scaled), scaling_applied, scaling_min, scaling_max


def load_image_rgb_with_metadata(path: str | Path) -> ImageLoadResult:
    """Load an image as an RGB uint8 NumPy array with image-load audit metadata.

    The first frame/page is used for multi-page TIFF files. EXIF orientation is
    respected. Alpha channels are composited onto a white background. Non-uint8
    numeric image arrays are min-max scaled to uint8 so thresholding always
    receives a consistent input type while recording the scaling range.
    """
    with Image.open(path) as img:
        page_count = _safe_page_count(img)

        original_format = str(img.format or "")
        original_mode = str(img.mode)
        original_dtype = _safe_array_dtype(img)
        frame_index = 0
        warnings: list[str] = []
        if page_count is not None and page_count > 1 and _is_tiff(path, original_format):
            warnings.append(
                f"Multi-frame TIFF contains {page_count} pages; only frame/page {frame_index} was analyzed."
            )

        # Read EXIF orientation while the source file pointer is intact, then
        # apply it to an in-memory frame copy. Calling ImageOps.exif_transpose()
        # directly on some multi-frame TIFF objects can require a stale file
        # pointer after array inspection, while copying first may drop the TIFF
        # orientation tag.
        exif_orientation = _safe_exif_orientation(img)
        resolution_metadata = _embedded_resolution_metadata(img, exif_orientation)
        img = _apply_exif_orientation(img.copy(), exif_orientation)
        scaling_applied = False
        scaling_min: float | None = None
        scaling_max: float | None = None

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img).convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8)
        else:
            arr = np.asarray(img)
            if arr.dtype != np.uint8 and np.issubdtype(arr.dtype, np.number) and arr.ndim in {2, 3}:
                rgb, scaling_applied, scaling_min, scaling_max = _load_numeric_array_as_rgb(
                    arr,
                    original_mode=original_mode,
                    original_dtype=original_dtype,
                    warnings=warnings,
                )
            elif img.mode in {"RGB", "L"}:
                rgb, scaling_applied, scaling_min, scaling_max = _load_numeric_array_as_rgb(
                    arr,
                    original_mode=original_mode,
                    original_dtype=original_dtype,
                    warnings=warnings,
                )
            else:
                # Pillow's RGB conversion handles palettes, CMYK, and most scanner TIFFs.
                img = img.convert("RGB")
                rgb = np.asarray(img, dtype=np.uint8)

        metadata = ImageLoadMetadata(
            original_format=original_format,
            original_mode=original_mode,
            original_dtype=original_dtype,
            frame_index=frame_index,
            page_count=page_count,
            scaling_applied=scaling_applied,
            scaling_min=scaling_min,
            scaling_max=scaling_max,
            **resolution_metadata,
            output_dtype=str(rgb.dtype),
            warnings=tuple(warnings),
        )
        return ImageLoadResult(rgb=rgb, metadata=metadata)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 NumPy array.

    The first frame/page is used for multi-page TIFF files. EXIF orientation is
    respected. Alpha channels are composited onto a white background.
    """
    return load_image_rgb_with_metadata(path).rgb


def save_image(array: np.ndarray, path: str | Path) -> None:
    """Save a uint8 image array using Pillow, creating parent folders."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(out)
