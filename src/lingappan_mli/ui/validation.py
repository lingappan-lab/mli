"""Input normalization and validation helpers for the Gradio UI."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from lingappan_mli.analysis import infer_slide_field
from lingappan_mli.contours import normalize_airspace_component_connectivity
from lingappan_mli.image_validation import (
    format_bytes,
    image_dimension_validation_errors as _image_dimension_validation_errors,
    image_file_validation,
)
from lingappan_mli.ui.config import (
    MAX_UPLOAD_COUNT,
    MAX_UPLOAD_DIMENSION,
    MAX_UPLOAD_FILE_BYTES,
    MAX_UPLOAD_PIXELS,
    MAX_UPLOAD_TOTAL_BYTES,
)



def image_dimension_validation_errors(
    path: Path,
    label: str,
    *,
    max_upload_dimension: int = MAX_UPLOAD_DIMENSION,
    max_upload_pixels: int = MAX_UPLOAD_PIXELS,
) -> list[str]:
    return _image_dimension_validation_errors(
        path,
        label,
        max_image_dimension=max_upload_dimension,
        max_image_pixels=max_upload_pixels,
    )


def upload_file_validation(
    raw_path: str,
    *,
    max_upload_file_bytes: int = MAX_UPLOAD_FILE_BYTES,
    max_upload_dimension: int = MAX_UPLOAD_DIMENSION,
    max_upload_pixels: int = MAX_UPLOAD_PIXELS,
) -> tuple[int, list[str]]:
    return image_file_validation(
        raw_path,
        max_image_file_bytes=max_upload_file_bytes,
        max_image_dimension=max_upload_dimension,
        max_image_pixels=max_upload_pixels,
    )


def upload_validation_errors(
    uploaded: list[str],
    *,
    max_upload_count: int = MAX_UPLOAD_COUNT,
    max_upload_file_bytes: int = MAX_UPLOAD_FILE_BYTES,
    max_upload_total_bytes: int = MAX_UPLOAD_TOTAL_BYTES,
    max_upload_pixels: int = MAX_UPLOAD_PIXELS,
    max_upload_dimension: int = MAX_UPLOAD_DIMENSION,
) -> list[str]:
    errors: list[str] = []
    if len(uploaded) > max_upload_count:
        errors.append(f"Upload at most {max_upload_count} images per run; {len(uploaded)} were selected.")

    total_bytes = 0
    for raw_path in uploaded:
        file_bytes, file_errors = upload_file_validation(
            raw_path,
            max_upload_file_bytes=max_upload_file_bytes,
            max_upload_dimension=max_upload_dimension,
            max_upload_pixels=max_upload_pixels,
        )
        total_bytes += file_bytes
        errors.extend(file_errors)

    if total_bytes > max_upload_total_bytes:
        errors.append(
            f"Selected files total {format_bytes(total_bytes)}, exceeding the batch limit "
            f"of {format_bytes(max_upload_total_bytes)}."
        )
    return errors


def display_upload_validation_errors(errors: list[str], max_items: int = 8) -> list[str]:
    if len(errors) <= max_items:
        return errors
    return [
        *errors[:max_items],
        f"...and {len(errors) - max_items} more upload validation issue(s).",
    ]


def normalize_files(files) -> list[str]:
    if files is None:
        return []
    if isinstance(files, (str, Path)):
        return [str(files)]
    normalized: list[str] = []
    for item in files:
        if isinstance(item, (str, Path)):
            normalized.append(str(item))
        elif hasattr(item, "name"):
            normalized.append(str(item.name))
        elif isinstance(item, dict) and "name" in item:
            normalized.append(str(item["name"]))
    return normalized


def combined_uploads(files, folders=None) -> list[str]:
    uploaded = normalize_files(files) + normalize_files(folders)
    return list(dict.fromkeys(uploaded))


def strategy_value(choice: str) -> str:
    normalized = (choice or "").lower()
    return "spacing" if "spacing" in normalized else "count"


def airspace_bright_value(choice) -> bool:
    if isinstance(choice, bool):
        return choice
    normalized = str(choice or "bright").lower()
    return normalized not in {"dark", "false", "0", "no"} and "dark" not in normalized


def component_connectivity_value(choice) -> int:
    if not str(choice or "").strip():
        return 8
    try:
        return normalize_airspace_component_connectivity(choice)
    except ValueError:
        raise ValueError("Airspace component connectivity must be 4 or 8.") from None


def infer_slide_id(path: str, separator: str) -> str:
    return infer_slide_field(Path(path).name, separator or "_")[0]


def infer_field_id(path: str, separator: str) -> str:
    return infer_slide_field(Path(path).name, separator or "_")[1]


def archive_stem(uploaded: list[str], slide_roi_separator: str) -> str:
    slide_ids = list(dict.fromkeys(infer_slide_id(path, slide_roi_separator) for path in uploaded))
    if not slide_ids:
        slide_part = "lingappan"
    elif len(slide_ids) == 1:
        slide_part = slide_ids[0]
    elif len(slide_ids) <= 3:
        slide_part = "_".join(slide_ids)
    else:
        slide_part = f"{slide_ids[0]}_plus_{len(slide_ids) - 1}_slides"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(f"{slide_part}_{len(uploaded)}-fields_mli_results_{timestamp}").stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "lingappan_mli_results"


def safe_float(value, fallback: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def safe_int(value, fallback: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return fallback
        return int(value)
    except (TypeError, ValueError):
        return fallback


def optional_int(value) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def metadata_text(value, fallback: str = "Not recorded") -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback
