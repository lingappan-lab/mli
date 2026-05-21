"""Reusable image preflight validation helpers."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image as PILImage


def _non_negative_int_env(name: str, default: int) -> int:
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


DEFAULT_MAX_IMAGE_COUNT = _non_negative_int_env("LINGAPPAN_MLI_MAX_UPLOAD_COUNT", 200)
DEFAULT_MAX_IMAGE_FILE_BYTES = _non_negative_int_env(
    "LINGAPPAN_MLI_MAX_UPLOAD_FILE_BYTES", 250 * 1024 * 1024
)
DEFAULT_MAX_IMAGE_TOTAL_BYTES = _non_negative_int_env(
    "LINGAPPAN_MLI_MAX_UPLOAD_TOTAL_BYTES", 2 * 1024 * 1024 * 1024
)
DEFAULT_MAX_IMAGE_PIXELS = _non_negative_int_env("LINGAPPAN_MLI_MAX_UPLOAD_PIXELS", 100_000_000)
DEFAULT_MAX_IMAGE_DIMENSION = _non_negative_int_env("LINGAPPAN_MLI_MAX_UPLOAD_DIMENSION", 50_000)


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(value, 0))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def image_dimension_validation_errors(
    path: Path,
    label: str,
    *,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> list[str]:
    """Return image metadata/dimension preflight errors for one file."""
    try:
        with PILImage.open(path) as image:
            width, height = image.size
    except PILImage.DecompressionBombError as exc:
        return [f"{label}: image metadata exceeds safety limits ({exc})."]
    except (OSError, ValueError) as exc:
        return [f"{label}: could not read image metadata ({exc})."]

    if width <= 0 or height <= 0:
        return [f"{label}: image dimensions are invalid ({width} × {height} px)."]

    errors: list[str] = []
    if width > max_image_dimension or height > max_image_dimension:
        errors.append(
            f"{label}: image dimensions {width} × {height} px exceed the maximum dimension "
            f"of {max_image_dimension:,} px."
        )
    pixels = width * height
    if pixels > max_image_pixels:
        errors.append(
            f"{label}: image has {pixels:,} pixels, exceeding the per-image limit "
            f"of {max_image_pixels:,} pixels."
        )
    return errors


def image_file_validation(
    raw_path: str | Path,
    *,
    max_image_file_bytes: int = DEFAULT_MAX_IMAGE_FILE_BYTES,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> tuple[int, list[str]]:
    """Return byte count and preflight errors for one image file path."""
    path = Path(raw_path)
    label = path.name or str(raw_path)
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        return 0, [f"{label}: file is not available ({exc})."]
    if not resolved.is_file():
        return 0, [f"{label}: input item is not a file."]

    try:
        file_bytes = resolved.stat().st_size
    except OSError as exc:
        return 0, [f"{label}: could not read file size ({exc})."]

    errors: list[str] = []
    if file_bytes > max_image_file_bytes:
        errors.append(
            f"{label}: file size {format_bytes(file_bytes)} exceeds the per-file limit "
            f"of {format_bytes(max_image_file_bytes)}."
        )
    errors.extend(
        image_dimension_validation_errors(
            resolved,
            label,
            max_image_dimension=max_image_dimension,
            max_image_pixels=max_image_pixels,
        )
    )
    return file_bytes, errors


def image_batch_validation_errors(
    image_paths: list[str | Path],
    *,
    max_image_count: int | None = DEFAULT_MAX_IMAGE_COUNT,
    max_image_file_bytes: int = DEFAULT_MAX_IMAGE_FILE_BYTES,
    max_image_total_bytes: int = DEFAULT_MAX_IMAGE_TOTAL_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
) -> list[str]:
    """Return preflight errors for a batch of image paths."""
    errors: list[str] = []
    if max_image_count is not None and len(image_paths) > max_image_count:
        errors.append(f"Process at most {max_image_count} images per run; {len(image_paths)} were selected.")

    total_bytes = 0
    for raw_path in image_paths:
        file_bytes, file_errors = image_file_validation(
            raw_path,
            max_image_file_bytes=max_image_file_bytes,
            max_image_dimension=max_image_dimension,
            max_image_pixels=max_image_pixels,
        )
        total_bytes += file_bytes
        errors.extend(file_errors)

    if total_bytes > max_image_total_bytes:
        errors.append(
            f"Selected files total {format_bytes(total_bytes)}, exceeding the batch limit "
            f"of {format_bytes(max_image_total_bytes)}."
        )
    return errors


def display_validation_errors(errors: list[str], max_items: int = 8) -> list[str]:
    if len(errors) <= max_items:
        return errors
    return [
        *errors[:max_items],
        f"...and {len(errors) - max_items} more image validation issue(s).",
    ]


def image_preflight_error_message(errors: list[str], *, max_items: int = 8) -> str:
    displayed = display_validation_errors(errors, max_items=max_items)
    bullet_list = "\n".join(f"- {error}" for error in displayed)
    return f"Image preflight failed:\n{bullet_list}"
