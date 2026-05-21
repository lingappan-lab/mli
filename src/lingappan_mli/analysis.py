"""High-level analysis pipeline for Lingappan MLI."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import secrets
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from .contours import (
    AREA_COLUMN_SEMANTICS,
    AREA_MEASUREMENT_METHOD,
    DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY,
    AirspaceContourAnalysis,
    airspace_component_connectivity_label,
    analyze_airspace_contours,
    area_measurement_method_description,
    normalize_airspace_component_connectivity,
)
from .excel import write_results_workbook
from .grid import GridLine, generate_grid_layout
from .image_validation import (
    DEFAULT_MAX_IMAGE_COUNT,
    DEFAULT_MAX_IMAGE_DIMENSION,
    DEFAULT_MAX_IMAGE_FILE_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_IMAGE_TOTAL_BYTES,
    image_batch_validation_errors,
    image_preflight_error_message,
)
from .io import ImageLoadMetadata, discover_images, load_image_rgb_with_metadata, safe_stem
from .measure import (
    measure_phase_chords,
    summarize_measurements,
    summarize_phase_chord_filter_sensitivity,
    summarize_phase_line_intercepts,
)
from .reporting import (
    CHORD_EXPORT_COLUMNS,
    _audit_excel_sheets,
    _compact_export_frame,
    _excel_sheets,
    _write_results_readme,
)
from .thresholding import threshold_airspace
from .visualization import (
    make_binary_chord_overlay,
    make_mask_overlay,
    make_overlay,
    make_particle_image,
    make_qc_panel,
    make_rejected_edge_chord_overlay,
    make_test_line_image,
    mask_to_uint8,
    save_pil,
)

LOGGER = logging.getLogger(__name__)

CHORD_COLUMNS = [
    "filename",
    "slide_id",
    "field_id",
    "measure_type",
    "phase",
    "orientation",
    "line_id",
    "line_position_px",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "length_px",
    "length_um",
    "edge_touching",
]

IMAGE_EXPORT_EXT = "png"
INPUT_IMAGE_TYPE = "pre_cropped_lung_roi"
FIELD_SELECTION_STAGE = "upstream_pre_app"
DEFAULT_PIXEL_WIDTH_UM = 0.57
DEFAULT_PIXEL_HEIGHT_UM = 0.57
DEFAULT_CALIBRATION_SOURCE = "default_0.57_um_per_px"
UNRECORDED_CALIBRATION_SOURCE = "manual_pixel_size_source_not_recorded"
DEFAULT_FIELD_SELECTION_METHOD = "not_recorded"
DEFAULT_FIELD_EXCLUSION_CRITERIA = "not_recorded"
EMBEDDED_CALIBRATION_REL_TOLERANCE = 0.02
EMBEDDED_CALIBRATION_COMPARE_MIN_UM_PER_PX = 0.001
EMBEDDED_CALIBRATION_COMPARE_MAX_UM_PER_PX = 100.0
BATCH_FIELD_SIZE_REL_TOLERANCE = 0.02
ProgressCallback = Callable[[float, str], None]


def _area_measurement_parameter_metadata(connectivity: int | str) -> dict[str, object]:
    return {
        "area_measurement_method": AREA_MEASUREMENT_METHOD,
        "area_measurement_connectivity": airspace_component_connectivity_label(connectivity),
        "area_column_semantics": AREA_COLUMN_SEMANTICS,
        "area_measurement_method_description": area_measurement_method_description(connectivity),
    }


def _area_measurement_summary_metadata(connectivity: int | str) -> dict[str, object]:
    normalized = normalize_airspace_component_connectivity(connectivity)
    return {
        "airspace_component_connectivity": int(normalized),
        "area_measurement_method": AREA_MEASUREMENT_METHOD,
        "area_measurement_connectivity": airspace_component_connectivity_label(normalized),
        "area_column_semantics": AREA_COLUMN_SEMANTICS,
    }


def _emit_progress(progress_callback: ProgressCallback | None, fraction: float, description: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, float(fraction))), description)


def _metadata_text(value: object, default: str = "") -> str:
    text = "" if value is None else str(value).strip()
    return text or default


def _is_default_pixel_size(pixel_width_um: float, pixel_height_um: float) -> bool:
    return math.isclose(float(pixel_width_um), DEFAULT_PIXEL_WIDTH_UM) and math.isclose(
        float(pixel_height_um), DEFAULT_PIXEL_HEIGHT_UM
    )


def _normalize_calibration_source(
    calibration_source: object,
    *,
    pixel_width_um: float,
    pixel_height_um: float,
) -> str:
    source = _metadata_text(calibration_source, DEFAULT_CALIBRATION_SOURCE)
    if source == DEFAULT_CALIBRATION_SOURCE and not _is_default_pixel_size(pixel_width_um, pixel_height_um):
        return UNRECORDED_CALIBRATION_SOURCE
    return source


def _uses_default_calibration(params: "AnalysisParams") -> bool:
    return params.calibration_source == DEFAULT_CALIBRATION_SOURCE and _is_default_pixel_size(
        params.pixel_width_um,
        params.pixel_height_um,
    )


def _calibration_warning(params: "AnalysisParams") -> str:
    if not _uses_default_calibration(params):
        return ""
    return (
        f"Default calibration is in use ({DEFAULT_PIXEL_WIDTH_UM:g} × {DEFAULT_PIXEL_HEIGHT_UM:g} µm/px). "
        "Verify pixel size against microscope/camera metadata or a scale bar before reporting measurements; "
        "record the source with calibration_source/--calibration-source when known."
    )


def _relative_difference(value: float, reference: float) -> float:
    denominator = max(abs(float(value)), abs(float(reference)), 1e-12)
    return abs(float(value) - float(reference)) / denominator


def _embedded_pixel_size_is_comparable(pixel_size_um: float | None) -> bool:
    return (
        pixel_size_um is not None
        and EMBEDDED_CALIBRATION_COMPARE_MIN_UM_PER_PX
        <= float(pixel_size_um)
        <= EMBEDDED_CALIBRATION_COMPARE_MAX_UM_PER_PX
    )


def _embedded_resolution_calibration_warning(
    image_load_metadata: ImageLoadMetadata,
    params: "AnalysisParams",
    *,
    rel_tolerance: float = EMBEDDED_CALIBRATION_REL_TOLERANCE,
) -> str:
    embedded_width = image_load_metadata.embedded_pixel_width_um
    embedded_height = image_load_metadata.embedded_pixel_height_um
    if not (
        _embedded_pixel_size_is_comparable(embedded_width)
        and _embedded_pixel_size_is_comparable(embedded_height)
    ):
        return ""

    width_difference = _relative_difference(embedded_width, params.pixel_width_um)
    height_difference = _relative_difference(embedded_height, params.pixel_height_um)
    if width_difference <= rel_tolerance and height_difference <= rel_tolerance:
        return ""

    source = image_load_metadata.embedded_resolution_source or "embedded image resolution metadata"
    unit = image_load_metadata.embedded_resolution_unit or "unknown unit"
    resolution_label = (
        f"{image_load_metadata.embedded_resolution_x:.6g} × "
        f"{image_load_metadata.embedded_resolution_y:.6g} px/{unit}"
        if image_load_metadata.embedded_resolution_x is not None
        and image_load_metadata.embedded_resolution_y is not None
        else f"unit={unit}"
    )
    return (
        f"Embedded image resolution metadata ({source}; {resolution_label}) implies pixel size "
        f"{embedded_width:.6g} × {embedded_height:.6g} µm/px, which differs from the analysis "
        f"calibration {params.pixel_width_um:.6g} × {params.pixel_height_um:.6g} µm/px "
        f"(relative differences {width_difference:.1%} width, {height_difference:.1%} height). "
        "DPI/resolution metadata can be generic or scanner-written; verify the calibration source "
        "before reporting physical measurements."
    )


def _image_load_metadata_with_calibration_warning(
    image_load_metadata: ImageLoadMetadata,
    params: "AnalysisParams",
) -> ImageLoadMetadata:
    warning = _embedded_resolution_calibration_warning(image_load_metadata, params)
    if not warning:
        return image_load_metadata
    return replace(image_load_metadata, warnings=(*image_load_metadata.warnings, warning))


def _pixel_area_um2(params: "AnalysisParams") -> float:
    return float(params.pixel_width_um * params.pixel_height_um)


def _calibration_audit_metadata(params: "AnalysisParams") -> dict[str, object]:
    return {
        "pixel_area_um2": _pixel_area_um2(params),
        "calibration_source": params.calibration_source,
        "calibration_is_default": bool(_uses_default_calibration(params)),
        "calibration_warning": _calibration_warning(params),
    }


@dataclass(frozen=True)
class AnalysisParams:
    """Configurable analysis parameters."""

    pixel_width_um: float = DEFAULT_PIXEL_WIDTH_UM
    pixel_height_um: float = DEFAULT_PIXEL_HEIGHT_UM
    grid_strategy: str = "count"  # "count" or "spacing"
    num_lines: int = 15
    line_spacing_um: float = 35.4
    grid_random_offset: bool = False
    grid_random_seed: int | None = None
    orientation: str = "both"  # "horizontal", "vertical", or "both"
    threshold_method: str = "huang"  # "huang" or "otsu"
    airspace_bright: bool = True
    airspace_component_connectivity: int = DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY
    exclude_edge_touching: bool = True
    min_chord_um: float = 0.0
    measure_non_airspace: bool = False
    slide_roi_separator: str = "_"
    input_image_type: str = INPUT_IMAGE_TYPE
    field_selection_stage: str = FIELD_SELECTION_STAGE
    field_selection_method: str = DEFAULT_FIELD_SELECTION_METHOD
    field_selection_notes: str = ""
    field_exclusion_criteria: str = DEFAULT_FIELD_EXCLUSION_CRITERIA
    calibration_source: str = DEFAULT_CALIBRATION_SOURCE

    def validated(self) -> "AnalysisParams":
        if self.pixel_width_um <= 0 or self.pixel_height_um <= 0:
            raise ValueError("Pixel width/height must be positive.")
        if self.num_lines < 1:
            raise ValueError("Number of lines must be at least 1.")
        if self.line_spacing_um <= 0:
            raise ValueError("Line spacing must be positive.")
        if self.min_chord_um < 0:
            raise ValueError("Minimum chord length cannot be negative.")
        airspace_component_connectivity = normalize_airspace_component_connectivity(
            self.airspace_component_connectivity
        )
        grid_random_seed = None if self.grid_random_seed is None else int(self.grid_random_seed)
        if grid_random_seed is not None and grid_random_seed < 0:
            raise ValueError("Grid random seed must be a non-negative integer.")
        if self.grid_random_offset and grid_random_seed is None:
            grid_random_seed = secrets.randbits(32)
        if self.input_image_type != INPUT_IMAGE_TYPE:
            raise ValueError(
                "input_image_type must be 'pre_cropped_lung_roi'; upload pre-cropped fields/ROIs, not whole-slide images."
            )
        if self.field_selection_stage != FIELD_SELECTION_STAGE:
            raise ValueError(
                "field_selection_stage must be 'upstream_pre_app'; this analyzer records upstream field selection but does not perform it."
            )
        calibration_source = _normalize_calibration_source(
            self.calibration_source,
            pixel_width_um=self.pixel_width_um,
            pixel_height_um=self.pixel_height_um,
        )
        return replace(
            self,
            grid_random_seed=grid_random_seed,
            airspace_component_connectivity=airspace_component_connectivity,
            calibration_source=calibration_source,
            field_selection_method=_metadata_text(self.field_selection_method, DEFAULT_FIELD_SELECTION_METHOD),
            field_selection_notes=_metadata_text(self.field_selection_notes),
            field_exclusion_criteria=_metadata_text(self.field_exclusion_criteria, DEFAULT_FIELD_EXCLUSION_CRITERIA),
        )


@dataclass
class AnalysisResult:
    """Result for one analyzed image."""

    summary: dict[str, object]
    chords: pd.DataFrame
    grid_lines: list[GridLine]
    output_dir: Path | None = None
    overlay_path: Path | None = None
    non_airspace_overlay_path: Path | None = None
    qc_panel_path: Path | None = None
    binary_path: Path | None = None
    final_contours_path: Path | None = None
    mask_overlay_path: Path | None = None
    mli_rejected_edge_overlay_path: Path | None = None
    non_airspace_rejected_edge_overlay_path: Path | None = None
    image_load_metadata: ImageLoadMetadata | None = None


def infer_slide_field(filename: str, separator: str = "_") -> tuple[str, str]:
    """Infer slide and field IDs from filename stem using the final separator."""
    stem = Path(filename).stem
    if separator and separator in stem:
        slide, field = stem.rsplit(separator, 1)
        return slide or stem, field or "field"
    return stem, "field"


def _offset_um(offset_px: float | None, pixel_size_um: float) -> float | None:
    return None if offset_px is None else float(offset_px * pixel_size_um)


def _field_audit_metadata(
    params: AnalysisParams,
    *,
    included_in_summary: bool,
    exclusion_flag: bool,
    exclusion_reason: str = "",
) -> dict[str, object]:
    """Return auditable per-field metadata for input/selection assumptions."""
    return {
        "input_image_type": params.input_image_type,
        "field_selection_stage": params.field_selection_stage,
        "field_selection_method": params.field_selection_method,
        "field_selection_notes": params.field_selection_notes,
        "field_exclusion_criteria": params.field_exclusion_criteria,
        "field_selected_by_app": False,
        "field_included_in_summary": bool(included_in_summary),
        "field_exclusion_flag": bool(exclusion_flag),
        "field_exclusion_reason": str(exclusion_reason or ""),
        "edge_exclusion_applied": bool(params.exclude_edge_touching),
        "short_chord_exclusion_applied": bool(params.min_chord_um > 0),
        **_calibration_audit_metadata(params),
        **_area_measurement_summary_metadata(params.airspace_component_connectivity),
    }


def _attach_image_metadata(chords: pd.DataFrame, filename: str, slide_id: str, field_id: str) -> pd.DataFrame:
    if chords.empty:
        return pd.DataFrame(columns=CHORD_COLUMNS)
    out = chords.copy()
    out.insert(0, "field_id", field_id)
    out.insert(0, "slide_id", slide_id)
    out.insert(0, "filename", filename)
    return out.reindex(columns=CHORD_COLUMNS)


def _measure_rejected_edge_phase(
    airspace_mask: np.ndarray,
    lines: list[GridLine],
    params: AnalysisParams,
    *,
    phase: str,
) -> pd.DataFrame:
    """Return min-length-passing chords rejected by the configured edge filter."""
    measure_columns = CHORD_COLUMNS[3:]
    if not params.exclude_edge_touching:
        return pd.DataFrame(columns=measure_columns)
    candidates = measure_phase_chords(
        airspace_mask,
        lines,
        phase=phase,
        pixel_width_um=params.pixel_width_um,
        pixel_height_um=params.pixel_height_um,
        exclude_edge_touching=False,
        min_chord_um=params.min_chord_um,
    )
    if candidates.empty:
        return pd.DataFrame(columns=measure_columns)
    rejected = candidates[candidates["edge_touching"].astype(bool)].copy()
    return rejected.reindex(columns=measure_columns)


def _save_line_detail_outputs(
    image_shape: tuple[int, int],
    airspace_mask: np.ndarray,
    lines: list[GridLine],
    chords: pd.DataFrame,
    out_dir: Path,
    *,
    measure_type: str,
) -> None:
    measure_key = "mli" if measure_type.lower() == "mli" else "non_airspace"
    measure_label = "MLI" if measure_key == "mli" else "Non-airspace"
    measure_folder = "03_mli" if measure_key == "mli" else "04_non_airspace"
    grids_dir = out_dir / measure_folder / "grids"
    overlays_dir = out_dir / measure_folder / "overlays"
    test_lines_dir = out_dir / measure_folder / "test_lines"
    grids_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    test_lines_dir.mkdir(parents=True, exist_ok=True)

    for orientation in ("Horizontal", "Vertical"):
        selected_lines = [line for line in lines if line.orientation == orientation]
        if not selected_lines:
            continue
        orientation_lower = orientation.lower()
        save_pil(
            make_test_line_image(image_shape, selected_lines, orientation_lower),
            test_lines_dir / f"{orientation_lower}_test_lines.{IMAGE_EXPORT_EXT}",
        )
        save_pil(
            make_particle_image(image_shape, chords, measure_type=measure_label, orientation=orientation),
            grids_dir / f"{orientation_lower}_grid_{measure_key}.{IMAGE_EXPORT_EXT}",
        )
        save_pil(
            make_binary_chord_overlay(airspace_mask, chords, measure_type=measure_label, orientation=orientation),
            overlays_dir / f"overlay_{orientation_lower}_{measure_key}.{IMAGE_EXPORT_EXT}",
        )


def _safe_float_for_title(value: object) -> float:
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return as_float if np.isfinite(as_float) else float("nan")


def _write_image_outputs(
    output_dir: str | Path,
    image_path: Path,
    filename: str,
    rgb: np.ndarray,
    gray: np.ndarray,
    airspace_mask: np.ndarray,
    contour_analysis: AirspaceContourAnalysis,
    lines: list[GridLine],
    chords: pd.DataFrame,
    summary: dict[str, object],
    params: AnalysisParams,
    progress_callback: ProgressCallback | None,
    output_folder_name: str | None = None,
) -> dict[str, Path | None]:
    out_dir = Path(output_dir) / (output_folder_name or safe_stem(image_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir = out_dir / "01_preprocessing"
    contours_dir = out_dir / "02_contours"
    data_dir = out_dir / "05_data"
    qc_dir = out_dir / "06_qc"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    contours_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Per-image chord data is stored once in 05_data to avoid duplicated
    # CSVs at the image-folder root. The primary CSVs are compact and
    # human-readable; *_audit.csv keeps the complete coordinate-level table.
    chords.to_csv(data_dir / "chords_audit.csv", index=False)
    _compact_export_frame(chords, CHORD_EXPORT_COLUMNS).to_csv(data_dir / "chords.csv", index=False)

    # Preprocessing and contour QC exports. PNG keeps these generated masks,
    # overlays, and panels lossless while avoiding very large TIFF exports.
    binary_path = preprocessing_dir / f"binary_airspace_mask.{IMAGE_EXPORT_EXT}"
    Image.fromarray(mask_to_uint8(airspace_mask)).save(binary_path)
    Image.fromarray(gray).save(preprocessing_dir / f"grayscale_8bit.{IMAGE_EXPORT_EXT}")

    mask_overlay_path = qc_dir / f"mask_overlay.{IMAGE_EXPORT_EXT}"
    save_pil(
        make_mask_overlay(
            rgb,
            airspace_mask,
            title=f"{filename} | analysis mask overlay",
        ),
        mask_overlay_path,
    )

    Image.fromarray(contour_analysis.all_contours_rgb).save(contours_dir / f"all_airspace_contours.{IMAGE_EXPORT_EXT}")
    final_contours_path = contours_dir / f"final_airspace_contours.{IMAGE_EXPORT_EXT}"
    Image.fromarray(contour_analysis.final_contours_rgb).save(final_contours_path)
    Image.fromarray(contour_analysis.final_contours_filled_rgb).save(
        contours_dir / f"final_airspace_contours_filled.{IMAGE_EXPORT_EXT}"
    )

    # Clinical overlays plus per-orientation line/chord detail outputs.
    mli_value = _safe_float_for_title(summary.get("mli_orientation_balanced_mean_um"))
    mli_title = f"{filename} | MLI={mli_value:.2f} µm | n={summary.get('mli_chord_count', 0)}"
    overlay = make_overlay(rgb, lines, chords, title=mli_title, measure_types={"MLI"})
    overlay_path = out_dir / "03_mli" / f"overlay_mli_combined.{IMAGE_EXPORT_EXT}"
    save_pil(overlay, overlay_path)
    mli_rejected_edge_chords = _measure_rejected_edge_phase(
        airspace_mask,
        lines,
        params,
        phase="airspace",
    )
    mli_rejected_edge_overlay_path = qc_dir / f"rejected_edge_chords_mli.{IMAGE_EXPORT_EXT}"
    save_pil(
        make_rejected_edge_chord_overlay(
            rgb,
            lines,
            mli_rejected_edge_chords,
            title=f"{filename} | rejected edge MLI chords={len(mli_rejected_edge_chords)}",
        ),
        mli_rejected_edge_overlay_path,
    )
    _save_line_detail_outputs(airspace_mask.shape, airspace_mask, lines, chords, out_dir, measure_type="MLI")

    non_airspace_overlay_path = None
    non_airspace_rejected_edge_overlay_path = None
    if params.measure_non_airspace:
        non_airspace_value = _safe_float_for_title(summary.get("mean_non_airspace_chord_um"))
        non_airspace_title = (
            f"{filename} | non-airspace chord={non_airspace_value:.2f} µm | "
            f"n={summary.get('non_airspace_chord_count', 0)}"
        )
        non_airspace_overlay = make_overlay(
            rgb,
            lines,
            chords,
            title=non_airspace_title,
            measure_types={"Non-airspace"},
        )
        non_airspace_overlay_path = out_dir / "04_non_airspace" / f"overlay_non_airspace_combined.{IMAGE_EXPORT_EXT}"
        save_pil(non_airspace_overlay, non_airspace_overlay_path)
        non_airspace_rejected_edge_chords = _measure_rejected_edge_phase(
            airspace_mask,
            lines,
            params,
            phase="non_airspace",
        )
        non_airspace_rejected_edge_overlay_path = qc_dir / f"rejected_edge_chords_non_airspace.{IMAGE_EXPORT_EXT}"
        save_pil(
            make_rejected_edge_chord_overlay(
                rgb,
                lines,
                non_airspace_rejected_edge_chords,
                title=f"{filename} | rejected edge non-airspace chords={len(non_airspace_rejected_edge_chords)}",
            ),
            non_airspace_rejected_edge_overlay_path,
        )
        _save_line_detail_outputs(
            airspace_mask.shape,
            airspace_mask,
            lines,
            chords,
            out_dir,
            measure_type="Non-airspace",
        )

    _emit_progress(progress_callback, 0.90, f"Building QC preview for {filename}")
    qc_panel = make_qc_panel(
        rgb,
        airspace_mask,
        overlay,
        title=f"Quality control: {filename}",
        final_airspaces_rgb=contour_analysis.final_contours_filled_rgb,
        edge_exclusion_applied=params.exclude_edge_touching,
    )
    qc_panel_path = out_dir / f"qc_panel.{IMAGE_EXPORT_EXT}"
    save_pil(qc_panel, qc_panel_path)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "output_dir": out_dir,
        "overlay_path": overlay_path,
        "non_airspace_overlay_path": non_airspace_overlay_path,
        "qc_panel_path": qc_panel_path,
        "binary_path": binary_path,
        "final_contours_path": final_contours_path,
        "mask_overlay_path": mask_overlay_path,
        "mli_rejected_edge_overlay_path": mli_rejected_edge_overlay_path,
        "non_airspace_rejected_edge_overlay_path": non_airspace_rejected_edge_overlay_path,
    }


def analyze_image(
    path: str | Path,
    params: AnalysisParams,
    output_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    output_folder_name: str | None = None,
) -> AnalysisResult:
    """Analyze one cropped field/ROI image.

    The input must be a pre-selected, pre-cropped lung field/ROI, not a
    whole-slide image. Upstream field-selection/exclusion metadata is recorded
    for auditability, but this function does not choose or validate ROIs.
    The default MLI measurement is the mean length of airspace chords along
    systematic test lines after Huang thresholding.
    """
    params = params.validated()
    image_path = Path(path)
    filename = image_path.name
    slide_id, field_id = infer_slide_field(filename, params.slide_roi_separator)

    _emit_progress(progress_callback, 0.03, f"Loading {filename}")
    image_load = load_image_rgb_with_metadata(image_path)
    rgb = image_load.rgb
    image_load_metadata = _image_load_metadata_with_calibration_warning(image_load.metadata, params)
    for warning in image_load_metadata.warnings:
        LOGGER.warning("%s: %s", filename, warning)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    pixel_area_um2 = _pixel_area_um2(params)
    field_width_um = float(width * params.pixel_width_um)
    field_height_um = float(height * params.pixel_height_um)

    _emit_progress(progress_callback, 0.14, f"Thresholding {filename}")
    gray, raw_airspace_mask, threshold_value = threshold_airspace(
        rgb,
        method=params.threshold_method,
        airspace_bright=params.airspace_bright,
    )
    airspace_mask = raw_airspace_mask
    airspace_fraction = float(np.mean(airspace_mask))

    _emit_progress(progress_callback, 0.28, f"Analyzing contours for {filename}")
    contour_analysis = analyze_airspace_contours(
        airspace_mask,
        pixel_width_um=params.pixel_width_um,
        pixel_height_um=params.pixel_height_um,
        exclude_edge_touching=params.exclude_edge_touching,
        connectivity=params.airspace_component_connectivity,
    )
    non_edge_airspace_fraction = float(
        contour_analysis.metrics["non_edge_airspace_component_area_pixels_mask"] / int(airspace_mask.size)
    )

    _emit_progress(progress_callback, 0.40, f"Generating test lines for {filename}")
    grid_field_seed = None
    if params.grid_random_offset and params.grid_random_seed is not None:
        seed_material = f"{int(params.grid_random_seed)}\0{filename}".encode("utf-8")
        digest = hashlib.blake2b(seed_material, digest_size=8).digest()
        grid_field_seed = int.from_bytes(digest, byteorder="little", signed=False) & 0xFFFFFFFF
    grid_layout = generate_grid_layout(
        airspace_mask.shape,
        orientation=params.orientation,
        strategy=params.grid_strategy,
        num_lines=params.num_lines,
        line_spacing_um=params.line_spacing_um,
        pixel_width_um=params.pixel_width_um,
        pixel_height_um=params.pixel_height_um,
        random_offset=params.grid_random_offset,
        random_seed=grid_field_seed,
    )
    lines = grid_layout.lines
    horizontal_lines = sum(1 for line in lines if line.orientation == "Horizontal")
    vertical_lines = sum(1 for line in lines if line.orientation == "Vertical")

    chord_measurement_kwargs = {
        "pixel_width_um": params.pixel_width_um,
        "pixel_height_um": params.pixel_height_um,
        "exclude_edge_touching": params.exclude_edge_touching,
        "min_chord_um": params.min_chord_um,
    }

    _emit_progress(progress_callback, 0.52, f"Measuring MLI chords for {filename}")
    mli_chords = measure_phase_chords(airspace_mask, lines, phase="airspace", **chord_measurement_kwargs)
    chord_frames = [_attach_image_metadata(mli_chords, filename, slide_id, field_id)]

    if params.measure_non_airspace:
        _emit_progress(progress_callback, 0.64, f"Measuring non-airspace chords for {filename}")
        non_airspace_chords = measure_phase_chords(
            airspace_mask,
            lines,
            phase="non_airspace",
            **chord_measurement_kwargs,
        )
        chord_frames.append(_attach_image_metadata(non_airspace_chords, filename, slide_id, field_id))

    _emit_progress(progress_callback, 0.72, f"Summarizing {filename}")
    non_empty_chord_frames = [frame for frame in chord_frames if not frame.empty]
    chords = (
        pd.concat(non_empty_chord_frames, ignore_index=True)
        if non_empty_chord_frames
        else pd.DataFrame(columns=CHORD_COLUMNS)
    )

    summary: dict[str, object] = {
        "filename": filename,
        "slide_id": slide_id,
        "field_id": field_id,
        **_field_audit_metadata(params, included_in_summary=True, exclusion_flag=False),
        "width_px": width,
        "height_px": height,
        **image_load_metadata.as_summary_dict(),
        "field_width_um": field_width_um,
        "field_height_um": field_height_um,
        "field_area_um2": float(field_width_um * field_height_um),
        "pixel_width_um": float(params.pixel_width_um),
        "pixel_height_um": float(params.pixel_height_um),
        "pixel_area_um2": pixel_area_um2,
        "threshold_method": params.threshold_method,
        "threshold_value": int(threshold_value),
        "airspace_fraction": airspace_fraction,
        "non_edge_airspace_fraction": non_edge_airspace_fraction,
        "grid_strategy": params.grid_strategy,
        "num_lines_requested": int(params.num_lines),
        "line_spacing_um_requested": float(params.line_spacing_um),
        "grid_random_offset": bool(params.grid_random_offset),
        "grid_random_seed": params.grid_random_seed,
        "grid_random_field_seed": grid_field_seed,
        "grid_horizontal_offset_px": grid_layout.horizontal_offset_px,
        "grid_vertical_offset_px": grid_layout.vertical_offset_px,
        "grid_horizontal_offset_um": _offset_um(grid_layout.horizontal_offset_px, params.pixel_height_um),
        "grid_vertical_offset_um": _offset_um(grid_layout.vertical_offset_px, params.pixel_width_um),
        "orientation": params.orientation,
        "horizontal_lines": int(horizontal_lines),
        "vertical_lines": int(vertical_lines),
        "exclude_edge_touching": bool(params.exclude_edge_touching),
        "min_chord_um": float(params.min_chord_um),
    }
    summary.update(contour_analysis.metrics)
    summary.update(summarize_measurements(chords, "MLI"))
    summary.update(
        summarize_phase_line_intercepts(
            airspace_mask,
            lines,
            phase="airspace",
            pixel_width_um=params.pixel_width_um,
            pixel_height_um=params.pixel_height_um,
            exclude_edge_touching=params.exclude_edge_touching,
            min_chord_um=params.min_chord_um,
            measure_type="MLI",
        )
    )
    if params.measure_non_airspace:
        summary.update(summarize_measurements(chords, "Non-airspace"))
        summary.update(
            summarize_phase_chord_filter_sensitivity(
                airspace_mask,
                lines,
                phase="non_airspace",
                pixel_width_um=params.pixel_width_um,
                pixel_height_um=params.pixel_height_um,
                exclude_edge_touching=params.exclude_edge_touching,
                min_chord_um=params.min_chord_um,
                measure_type="Non-airspace",
            )
        )

    output_paths: dict[str, Path | None] = {}
    if output_dir is not None:
        _emit_progress(progress_callback, 0.78, f"Writing outputs for {filename}")
        output_paths = _write_image_outputs(
            output_dir,
            image_path,
            filename,
            rgb,
            gray,
            airspace_mask,
            contour_analysis,
            lines,
            chords,
            summary,
            params,
            progress_callback,
            output_folder_name=output_folder_name,
        )

    _emit_progress(progress_callback, 1.0, f"Completed {filename}")
    return AnalysisResult(
        summary=summary,
        chords=chords,
        grid_lines=lines,
        **output_paths,
        image_load_metadata=image_load_metadata,
    )


def _finite_numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values for one column, dropping missing/non-finite values."""
    if column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[np.isfinite(values)]


def _first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first candidate column present in a DataFrame."""
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _field_sample_sd(frame: pd.DataFrame, column: str) -> float | None:
    """Sample SD across field-level values, returning 0 for a single measured field."""
    values = _finite_numeric_values(frame, column)
    count = int(len(values))
    if count == 0:
        return None
    if count == 1:
        return 0.0
    return float(values.std(ddof=1))


def _field_sem(frame: pd.DataFrame, column: str) -> float | None:
    """SEM across field-level values; fields, not chords, are the sampling units."""
    values = _finite_numeric_values(frame, column)
    count = int(len(values))
    if count == 0:
        return None
    if count == 1:
        return 0.0
    sd = float(values.std(ddof=1))
    return float(sd / math.sqrt(count))


def _chord_pooled_mean(frame: pd.DataFrame, value_column: str, count_column: str) -> float | None:
    """Mean across all accepted chords in a slide, weighting fields by chord count."""
    if value_column not in frame.columns or count_column not in frame.columns:
        return None
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[count_column], errors="coerce")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not bool(valid.any()):
        return None
    denominator = float(weights[valid].sum())
    if denominator <= 0:
        return None
    return float((values[valid] * weights[valid]).sum() / denominator)


def _mean_chords_per_field(frame: pd.DataFrame, count_column: str) -> float | None:
    """Mean accepted chord count per analyzed field for a slide."""
    if count_column not in frame.columns or len(frame) == 0:
        return None
    counts = pd.to_numeric(frame[count_column], errors="coerce").fillna(0.0)
    return float(counts.sum() / len(frame))


SlideMetricSpec = dict[str, list[str] | str]


def _blank_slide_metric(
    record: dict[str, object],
    spec: SlideMetricSpec,
    measured_field_count: int = 0,
) -> None:
    """Populate blank values for one slide-level metric family."""
    record[spec["measured_field_count_column"]] = measured_field_count
    for column_key in (
        "field_balanced_column",
        "field_sd_column",
        "field_sem_column",
        "sem_alias_column",
        "chord_pooled_column",
    ):
        record[spec[column_key]] = None


def _slide_metric_spec(
    *,
    column_stem: str,
    chords_stem: str,
    field_value_candidates: list[str],
    pooled_value_candidates: list[str],
    count_candidates: list[str],
    measured_stem: str | None = None,
) -> SlideMetricSpec:
    measured_prefix = measured_stem or column_stem
    return {
        "field_value_candidates": field_value_candidates,
        "pooled_value_candidates": pooled_value_candidates,
        "count_candidates": count_candidates,
        "field_balanced_column": f"field_balanced_mean_{column_stem}_um",
        "field_sd_column": f"field_sd_{column_stem}_um",
        "field_sem_column": f"field_sem_{column_stem}_um",
        "sem_alias_column": f"sem_{column_stem}_um",
        "chord_pooled_column": f"chord_pooled_mean_{column_stem}_um",
        "mean_chords_column": f"mean_{chords_stem}_chords_per_field",
        "measured_field_count_column": f"{measured_prefix}_measured_field_count",
    }


SLIDE_WEIGHTING_METRIC_SPECS = [
    _slide_metric_spec(
        column_stem="mli",
        chords_stem="mli",
        field_value_candidates=[
            "mli_orientation_balanced_direct_mean_um",
            "mli_direct_orientation_balanced_mean_um",
            "mli_orientation_balanced_mean_um",
        ],
        pooled_value_candidates=["mli_pooled_chord_mean_um", "mli_mean_um"],
        count_candidates=["mli_chord_count", "mli_chord_count_accepted"],
    ),
    _slide_metric_spec(
        column_stem="non_airspace_chord",
        chords_stem="non_airspace",
        measured_stem="non_airspace",
        field_value_candidates=[
            "mean_non_airspace_chord_um",
            "orientation_balanced_mean_non_airspace_chord_um",
            "non_airspace_orientation_balanced_direct_mean_um",
            "non_airspace_direct_orientation_balanced_mean_um",
            "non_airspace_orientation_balanced_mean_um",
        ],
        pooled_value_candidates=[
            "pooled_mean_non_airspace_chord_um",
            "non_airspace_pooled_chord_mean_um",
            "non_airspace_mean_um",
        ],
        count_candidates=[
            "non_airspace_chord_count",
            "non_airspace_chord_count_accepted",
        ],
    ),
]


def _slide_weighting_record(slide_id: object, group: pd.DataFrame) -> dict[str, object]:
    """Return explicit field-balanced and chord-pooled slide summary aliases."""
    record: dict[str, object] = {"slide_id": slide_id}
    metric_specs = SLIDE_WEIGHTING_METRIC_SPECS
    for spec in metric_specs:
        field_value_column = _first_existing_column(group, spec["field_value_candidates"])
        pooled_value_column = _first_existing_column(group, spec["pooled_value_candidates"])
        count_column = _first_existing_column(group, spec["count_candidates"])
        if count_column is None:
            continue
        record[spec["mean_chords_column"]] = _mean_chords_per_field(group, count_column)
        if field_value_column is None:
            _blank_slide_metric(record, spec)
            continue
        field_values = _finite_numeric_values(group, field_value_column)
        if field_values.empty:
            _blank_slide_metric(record, spec)
            continue
        record[spec["measured_field_count_column"]] = int(len(field_values))
        field_sem = _field_sem(group, field_value_column)
        record[spec["field_balanced_column"]] = float(field_values.mean())
        record[spec["field_sd_column"]] = _field_sample_sd(group, field_value_column)
        record[spec["field_sem_column"]] = field_sem
        record[spec["sem_alias_column"]] = field_sem
        if pooled_value_column is not None:
            record[spec["chord_pooled_column"]] = _chord_pooled_mean(
                group,
                pooled_value_column,
                count_column,
            )
        else:
            record[spec["chord_pooled_column"]] = None
    return record


def _add_slide_weighting_columns(
    slide_summary: pd.DataFrame,
    field_summary_for_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Append explicit weighting/SEM slide-summary columns without changing existing columns."""
    records = [
        _slide_weighting_record(slide_id, group)
        for slide_id, group in field_summary_for_agg.groupby("slide_id", dropna=False)
    ]
    weighting = pd.DataFrame(records)
    if weighting.empty or weighting.columns.tolist() == ["slide_id"]:
        return slide_summary
    return slide_summary.merge(weighting, on="slide_id", how="left")


def _apply_single_field_sd_semantics(slide_summary: pd.DataFrame) -> pd.DataFrame:
    """Set slide-level sample SD to 0 only when exactly one field has a valid measurement."""
    sd_count_columns = (
        ("sd_mli_um", "mli_measured_field_count"),
        ("sd_non_airspace_chord_um", "non_airspace_measured_field_count"),
    )
    for sd_column, count_column in sd_count_columns:
        if sd_column not in slide_summary.columns or count_column not in slide_summary.columns:
            continue
        counts = pd.to_numeric(slide_summary[count_column], errors="coerce")
        missing_sd = pd.to_numeric(slide_summary[sd_column], errors="coerce").isna()
        slide_summary.loc[(counts == 1) & missing_sd, sd_column] = 0.0
    return slide_summary


def _add_available_aggregations(
    aggregations: dict[str, tuple[str, str]],
    frame: pd.DataFrame,
    specs: Iterable[tuple[str, str, str]],
) -> None:
    """Add aggregation specs whose source columns exist in the field summary."""
    for output_column, source_column, aggregation_name in specs:
        if source_column in frame.columns:
            aggregations[output_column] = (source_column, aggregation_name)


def summarize_by_slide(field_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate field-level summaries by slide, preserving field-balanced and chord-pooled views."""
    if field_summary.empty or "slide_id" not in field_summary.columns:
        return pd.DataFrame()

    aggregations: dict[str, tuple[str, str]] = {
        "field_count": ("filename", "count"),
        "mean_airspace_fraction": ("airspace_fraction", "mean"),
        "mean_total_airspace_area_um2_mask": ("total_airspace_area_um2_mask", "mean"),
        "mean_final_airspace_area_um2_mask": ("final_airspace_area_um2_mask", "mean"),
        "mean_non_edge_airspace_component_area_um2_mask": (
            "non_edge_airspace_component_area_um2_mask",
            "mean",
        ),
        "mean_largest_airspace_component_area_um2": ("largest_airspace_component_area_um2", "mean"),
        "mean_largest_airspace_component_fraction_of_airspace": (
            "largest_airspace_component_fraction_of_airspace",
            "mean",
        ),
        "mean_mli_um": ("mli_orientation_balanced_mean_um", "mean"),
        "median_mli_um": ("mli_orientation_balanced_mean_um", "median"),
        "sd_mli_um": ("mli_orientation_balanced_mean_um", "std"),
        "total_mli_chords": ("mli_chord_count", "sum"),
    }
    _add_available_aggregations(
        aggregations,
        field_summary,
        [("mean_non_edge_airspace_fraction", "non_edge_airspace_fraction", "mean")],
    )
    _add_available_aggregations(
        aggregations,
        field_summary,
        [
            ("airspace_component_connectivity", "airspace_component_connectivity", "first"),
            ("area_measurement_connectivity", "area_measurement_connectivity", "first"),
            ("calibration_source", "calibration_source", "first"),
            ("calibration_is_default", "calibration_is_default", "first"),
            ("pixel_area_um2", "pixel_area_um2", "first"),
            ("mean_field_width_um", "field_width_um", "mean"),
            ("mean_field_height_um", "field_height_um", "mean"),
            ("mean_field_area_um2", "field_area_um2", "mean"),
        ],
    )
    _add_available_aggregations(
        aggregations,
        field_summary,
        [
            ("mean_mli_direct_um", "mli_direct_orientation_balanced_mean_um", "mean"),
            ("mean_mli_orientation_balanced_direct_um", "mli_orientation_balanced_direct_mean_um", "mean"),
            ("mean_mli_pooled_chord_um", "mli_pooled_chord_mean_um", "mean"),
            ("mean_mli_horizontal_vertical_mean_ratio", "mli_direct_horizontal_vertical_mean_ratio", "mean"),
            ("mean_mli_horizontal_vertical_mean_delta_um", "mli_direct_horizontal_vertical_mean_delta_um", "mean"),
            ("mean_mli_indirect_equivalent_um", "mli_indirect_equivalent_orientation_balanced_mean_um", "mean"),
            ("mean_mli_raw_um", "mli_orientation_balanced_mean_um_raw", "mean"),
            ("mean_mli_include_edge_um", "mli_orientation_balanced_mean_um_include_edge", "mean"),
            ("mean_mli_edge_excluded_um", "mli_orientation_balanced_mean_um_edge_excluded", "mean"),
            ("mean_mli_accepted_um", "mli_orientation_balanced_mean_um_accepted", "mean"),
            ("mean_mli_chord_fraction_min_length_excluded", "mli_chord_fraction_min_length_excluded", "mean"),
            ("mean_mli_chord_fraction_edge_excluded", "mli_chord_fraction_edge_excluded", "mean"),
            ("total_mli_chords_raw", "mli_chord_count_raw", "sum"),
            ("total_mli_chords_include_edge", "mli_chord_count_include_edge", "sum"),
            ("total_mli_chords_min_length_excluded", "mli_chord_count_min_length_excluded", "sum"),
            ("total_mli_chords_edge_excluded", "mli_chord_count_edge_excluded", "sum"),
            ("total_mli_chords_accepted", "mli_chord_count_accepted", "sum"),
            ("total_mli_airspace_line_length_um_raw", "mli_total_airspace_line_length_um_raw", "sum"),
            ("total_mli_airspace_line_length_um_accepted", "mli_total_airspace_line_length_um_accepted", "sum"),
            (
                "total_mli_airspace_boundary_intersections_raw",
                "mli_airspace_boundary_intersection_count_raw",
                "sum",
            ),
            (
                "total_mli_airspace_boundary_intersections_accepted",
                "mli_airspace_boundary_intersection_count_accepted",
                "sum",
            ),
        ],
    )
    non_airspace_count_columns = (
        "non_airspace_chord_count",
        "non_airspace_chord_count_include_edge",
        "non_airspace_chord_count_raw",
    )
    has_non_airspace_measurements = (
        "non_airspace_orientation_balanced_mean_um" in field_summary.columns
        or "mean_non_airspace_chord_um" in field_summary.columns
        or any(column in field_summary.columns for column in non_airspace_count_columns)
    )
    if has_non_airspace_measurements:
        _add_available_aggregations(
            aggregations,
            field_summary,
            [
                ("mean_non_airspace_chord_um", "mean_non_airspace_chord_um", "mean"),
                ("median_non_airspace_chord_um", "mean_non_airspace_chord_um", "median"),
                ("sd_non_airspace_chord_um", "mean_non_airspace_chord_um", "std"),
                (
                    "mean_orientation_balanced_non_airspace_chord_um",
                    "orientation_balanced_mean_non_airspace_chord_um",
                    "mean",
                ),
                ("mean_pooled_non_airspace_chord_um", "pooled_mean_non_airspace_chord_um", "mean"),
                ("total_non_airspace_chords", "non_airspace_chord_count", "sum"),
                (
                    "mean_non_airspace_direct_um",
                    "non_airspace_orientation_balanced_direct_mean_um",
                    "mean",
                ),
                ("mean_non_airspace_pooled_chord_um", "non_airspace_pooled_chord_mean_um", "mean"),
                (
                    "mean_non_airspace_horizontal_vertical_mean_ratio",
                    "non_airspace_direct_horizontal_vertical_mean_ratio",
                    "mean",
                ),
                (
                    "mean_non_airspace_horizontal_vertical_mean_delta_um",
                    "non_airspace_direct_horizontal_vertical_mean_delta_um",
                    "mean",
                ),
                ("mean_non_airspace_raw_um", "non_airspace_orientation_balanced_mean_um_raw", "mean"),
                (
                    "mean_non_airspace_include_edge_um",
                    "non_airspace_orientation_balanced_mean_um_include_edge",
                    "mean",
                ),
                (
                    "mean_non_airspace_edge_excluded_um",
                    "non_airspace_orientation_balanced_mean_um_edge_excluded",
                    "mean",
                ),
                ("mean_non_airspace_accepted_um", "non_airspace_orientation_balanced_mean_um_accepted", "mean"),
                (
                    "mean_non_airspace_chord_fraction_min_length_excluded",
                    "non_airspace_chord_fraction_min_length_excluded",
                    "mean",
                ),
                (
                    "mean_non_airspace_chord_fraction_edge_excluded",
                    "non_airspace_chord_fraction_edge_excluded",
                    "mean",
                ),
                ("total_non_airspace_chords_raw", "non_airspace_chord_count_raw", "sum"),
                ("total_non_airspace_chords_include_edge", "non_airspace_chord_count_include_edge", "sum"),
                (
                    "total_non_airspace_chords_min_length_excluded",
                    "non_airspace_chord_count_min_length_excluded",
                    "sum",
                ),
                ("total_non_airspace_chords_edge_excluded", "non_airspace_chord_count_edge_excluded", "sum"),
                ("total_non_airspace_chords_accepted", "non_airspace_chord_count_accepted", "sum"),
            ],
        )

    field_summary_for_agg = field_summary.copy()
    numeric_aggregation_columns = {
        source_column
        for source_column, aggregation_name in aggregations.values()
        if source_column != "filename" and aggregation_name in {"mean", "median", "std", "sum"}
    }
    for column in numeric_aggregation_columns:
        if column in field_summary_for_agg.columns:
            field_summary_for_agg[column] = pd.to_numeric(field_summary_for_agg[column], errors="coerce")

    slide_summary = field_summary_for_agg.groupby("slide_id", dropna=False).agg(**aggregations).reset_index()
    slide_summary = _add_slide_weighting_columns(slide_summary, field_summary_for_agg)
    slide_summary = _apply_single_field_sd_semantics(slide_summary)
    return slide_summary


INTERNAL_RESULT_COLUMNS = {"_input_index", "_input_path"}
WORKBOOK_EXPORT_FILENAMES = {"lingappan_mli_results.xlsx", "lingappan_mli_audit.xlsx"}


def _drop_internal_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove app-only row identity columns before writing user/audit exports."""
    internal_columns = [column for column in INTERNAL_RESULT_COLUMNS if column in frame.columns]
    return frame.drop(columns=internal_columns) if internal_columns else frame


def _unique_output_folder_names(images: Iterable[Path]) -> dict[Path, str]:
    """Return collision-free per-image folder names while preserving unique stems."""
    image_list = list(images)
    base_names = {image: safe_stem(image) for image in image_list}
    counts = pd.Series(list(base_names.values()), dtype="object").value_counts().to_dict()
    unique_bases = {base for base, count in counts.items() if count == 1}
    used = set(unique_bases)
    output_names: dict[Path, str] = {}

    for image in image_list:
        base_name = base_names[image]
        if counts[base_name] == 1:
            output_names[image] = base_name
            continue
        if base_name not in used:
            output_names[image] = base_name
            used.add(base_name)
            continue
        suffix = 2
        while f"{base_name}__{suffix}" in used:
            suffix += 1
        output_name = f"{base_name}__{suffix}"
        output_names[image] = output_name
        used.add(output_name)

    return output_names


def _workbook_export_warning_row(
    filename: str,
    exc: Exception,
    params: AnalysisParams,
) -> dict[str, object]:
    message = f"Failed to write {filename}: {exc}"
    LOGGER.warning(message)
    return {
        "filename": filename,
        "status": "warning",
        "message": message,
        "traceback": traceback.format_exc(),
        **_field_audit_metadata(params, included_in_summary=False, exclusion_flag=False),
    }


def _log_export_frames_from_rows(logs: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    log_df = pd.DataFrame(logs)
    log_df_export = _drop_internal_result_columns(log_df)
    log_df_public = log_df_export.copy()
    if "_input_index" in log_df:
        log_df_public.attrs["_input_index"] = log_df["_input_index"].tolist()
    return log_df_export, log_df_public


def _write_workbook_and_refresh_logs(
    path: Path,
    sheets: dict[str, pd.DataFrame],
    logs: list[dict[str, object]],
    params: AnalysisParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        write_results_workbook(path, sheets)
    except Exception as exc:
        logs.append(_workbook_export_warning_row(path.name, exc, params))
    return _log_export_frames_from_rows(logs)


def _format_dimension_counts(counts: pd.Series, *, max_items: int = 6) -> str:
    parts: list[str] = []
    for index, count in counts.head(max_items).items():
        width, height = index
        parts.append(f"{int(width)}×{int(height)} px (n={int(count)})")
    if len(counts) > max_items:
        parts.append(f"…and {len(counts) - max_items} more size(s)")
    return ", ".join(parts)


def _relative_span(values: pd.Series) -> tuple[float, float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return 0.0, math.nan, math.nan, math.nan
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    median = float(numeric.median())
    span = maximum - minimum
    relative = span / max(abs(median), 1e-12)
    return float(relative), minimum, maximum, median


def _batch_qc_warning_row(message: str, params: AnalysisParams) -> dict[str, object]:
    LOGGER.warning(message)
    return {
        "filename": "batch_qc",
        "status": "warning",
        "message": message,
        "traceback": "",
        "_input_index": 0,
        "_input_path": "",
        **_field_audit_metadata(params, included_in_summary=False, exclusion_flag=False),
    }


def _batch_field_consistency_warning_rows(
    field_summary: pd.DataFrame,
    params: AnalysisParams,
    *,
    rel_tolerance: float = BATCH_FIELD_SIZE_REL_TOLERANCE,
) -> list[dict[str, object]]:
    if len(field_summary) <= 1:
        return []

    warnings: list[dict[str, object]] = []
    if {"width_px", "height_px"}.issubset(field_summary.columns):
        dimensions = field_summary[["width_px", "height_px"]].copy()
        dimensions["width_px"] = pd.to_numeric(dimensions["width_px"], errors="coerce")
        dimensions["height_px"] = pd.to_numeric(dimensions["height_px"], errors="coerce")
        dimensions = dimensions.dropna()
        if not dimensions.empty:
            dimensions = dimensions.astype({"width_px": int, "height_px": int})
            counts = dimensions.groupby(["width_px", "height_px"]).size().sort_values(ascending=False)
            if len(counts) > 1:
                message = (
                    "Batch contains multiple analyzed image pixel dimensions: "
                    f"{_format_dimension_counts(counts)}. Confirm fields were cropped/resampled "
                    "consistently before comparing field or slide summaries."
                )
                warnings.append(_batch_qc_warning_row(message, params))

    if {"field_width_um", "field_height_um"}.issubset(field_summary.columns):
        width_rel, width_min, width_max, _ = _relative_span(field_summary["field_width_um"])
        height_rel, height_min, height_max, _ = _relative_span(field_summary["field_height_um"])
        if width_rel > rel_tolerance or height_rel > rel_tolerance:
            message = (
                f"Batch physical field size varies by more than {rel_tolerance:.1%} after applying "
                f"calibration: width {width_min:.6g}–{width_max:.6g} µm "
                f"({width_rel:.1%} span), height {height_min:.6g}–{height_max:.6g} µm "
                f"({height_rel:.1%} span). Confirm all fields use the same magnification, "
                "pixel-size calibration, and crop protocol before comparing summaries."
            )
            warnings.append(_batch_qc_warning_row(message, params))

    return warnings


def _workbook_export_warnings(logs: list[dict[str, object]]) -> list[tuple[str, str]]:
    return [
        (str(row.get("filename", "")), str(row.get("message", "")))
        for row in logs
        if row.get("status") == "warning" and row.get("filename") in WORKBOOK_EXPORT_FILENAMES
    ]


def process_files(
    paths: Iterable[str | Path],
    output_dir: str | Path,
    params: AnalysisParams,
    progress_callback: ProgressCallback | None = None,
    *,
    validate_inputs: bool = True,
    max_image_count: int | None = DEFAULT_MAX_IMAGE_COUNT,
    max_image_file_bytes: int = DEFAULT_MAX_IMAGE_FILE_BYTES,
    max_image_total_bytes: int = DEFAULT_MAX_IMAGE_TOTAL_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
) -> dict[str, object]:
    """Process image files/directories and write a complete results folder."""
    params = params.validated()
    images = discover_images(paths)
    if not images:
        raise ValueError("No supported images found. Upload or provide TIF/TIFF/PNG/JPG/BMP images.")
    if validate_inputs:
        preflight_errors = image_batch_validation_errors(
            images,
            max_image_count=max_image_count,
            max_image_file_bytes=max_image_file_bytes,
            max_image_total_bytes=max_image_total_bytes,
            max_image_pixels=max_image_pixels,
            max_image_dimension=max_image_dimension,
        )
        if preflight_errors:
            raise ValueError(image_preflight_error_message(preflight_errors))

    _emit_progress(progress_callback, 0.0, f"Discovered {len(images)} image file(s)")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    field_rows: list[dict[str, object]] = []
    chord_frames: list[pd.DataFrame] = []
    preview_paths: list[Path] = []
    logs: list[dict[str, object]] = []
    output_folder_names = _unique_output_folder_names(images)

    calibration_warning = _calibration_warning(params)
    pixel_area_um2 = _pixel_area_um2(params)
    calibration_is_default = _uses_default_calibration(params)
    if calibration_warning:
        LOGGER.warning(calibration_warning)
        logs.append(
            {
                "filename": "calibration",
                "status": "warning",
                "message": calibration_warning,
                "traceback": "",
                "_input_index": 0,
                "_input_path": "",
                **_field_audit_metadata(params, included_in_summary=False, exclusion_flag=False),
            }
        )

    total_images = len(images)
    image_progress_span = 0.86
    for image_index, image_path in enumerate(images, start=1):
        def image_progress(fraction: float, description: str) -> None:
            completed_before_image = image_index - 1
            image_fraction = (completed_before_image + fraction) / total_images
            _emit_progress(
                progress_callback,
                image_fraction * image_progress_span,
                f"{description} ({image_index}/{total_images})",
            )

        try:
            result = analyze_image(
                image_path,
                params,
                out,
                progress_callback=image_progress,
                output_folder_name=output_folder_names[image_path],
            )
            field_rows.append(
                {
                    **result.summary,
                    "_input_index": image_index,
                    "_input_path": str(image_path),
                }
            )
            if not result.chords.empty:
                chord_frames.append(result.chords)
            if result.qc_panel_path is not None:
                preview_paths.append(result.qc_panel_path)
            image_load_metadata = result.image_load_metadata
            assert image_load_metadata is not None
            image_load_log_metadata = image_load_metadata.as_summary_dict()
            logs.append(
                {
                    "filename": image_path.name,
                    "status": "ok",
                    "message": "",
                    "traceback": "",
                    "_input_index": image_index,
                    "_input_path": str(image_path),
                    **_field_audit_metadata(params, included_in_summary=True, exclusion_flag=False),
                    **image_load_log_metadata,
                }
            )
            for warning in image_load_metadata.warnings:
                logs.append(
                    {
                        "filename": image_path.name,
                        "status": "warning",
                        "message": warning,
                        "traceback": "",
                        "_input_index": image_index,
                        "_input_path": str(image_path),
                        **_field_audit_metadata(params, included_in_summary=True, exclusion_flag=False),
                        **image_load_log_metadata,
                    }
                )
        except Exception as exc:  # Keep batch processing resilient.
            logs.append(
                {
                    "filename": image_path.name,
                    "status": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "_input_index": image_index,
                    "_input_path": str(image_path),
                    **_field_audit_metadata(
                        params,
                        included_in_summary=False,
                        exclusion_flag=True,
                        exclusion_reason=f"processing_error: {exc}",
                    ),
                }
            )
            _emit_progress(
                progress_callback,
                image_index / total_images * image_progress_span,
                f"Recorded error for {image_path.name} ({image_index}/{total_images})",
            )

    _emit_progress(progress_callback, 0.90, "Preparing summary workbook and chord tables")
    field_summary = pd.DataFrame(field_rows)
    logs.extend(_batch_field_consistency_warning_rows(field_summary, params))
    field_summary_export = _drop_internal_result_columns(field_summary)
    all_chords = pd.concat(chord_frames, ignore_index=True) if chord_frames else pd.DataFrame(columns=CHORD_COLUMNS)
    slide_summary = summarize_by_slide(field_summary_export)
    log_df_export, log_df_public = _log_export_frames_from_rows(logs)
    field_summary_public = field_summary_export.copy()
    if "_input_index" in field_summary:
        field_summary_public.attrs["_input_index"] = field_summary["_input_index"].tolist()

    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    all_chords.to_csv(audit_dir / "all_chords_audit.csv", index=False)
    _compact_export_frame(all_chords, CHORD_EXPORT_COLUMNS).to_csv(out / "all_chords.csv", index=False)

    parameter_record = {
        **asdict(params),
        **_calibration_audit_metadata(params),
        **_area_measurement_parameter_metadata(params.airspace_component_connectivity),
    }
    with open(out / "parameters.json", "w", encoding="utf-8") as handle:
        json.dump(parameter_record, handle, indent=2)
    _write_results_readme(
        out,
        params,
        pixel_area_um2=pixel_area_um2,
        calibration_warning=calibration_warning,
    )

    _emit_progress(progress_callback, 0.96, "Writing compact Excel workbook")
    log_df_export, log_df_public = _write_workbook_and_refresh_logs(
        out / "lingappan_mli_results.xlsx",
        _excel_sheets(
            field_summary_export,
            slide_summary,
            all_chords,
            log_df_export,
            params,
            pixel_area_um2=pixel_area_um2,
            calibration_is_default=calibration_is_default,
            calibration_warning=calibration_warning,
        ),
        logs,
        params,
    )

    _emit_progress(progress_callback, 0.98, "Writing audit Excel workbook")
    log_df_export, log_df_public = _write_workbook_and_refresh_logs(
        audit_dir / "lingappan_mli_audit.xlsx",
        _audit_excel_sheets(field_summary_export, slide_summary, log_df_export, parameter_record),
        logs,
        params,
    )

    workbook_warnings = _workbook_export_warnings(logs)
    if workbook_warnings:
        _write_results_readme(
            out,
            params,
            pixel_area_um2=pixel_area_um2,
            calibration_warning=calibration_warning,
            export_warnings=workbook_warnings,
        )
        log_df_export.to_csv(audit_dir / "processing_log.csv", index=False)

    _emit_progress(progress_callback, 1.0, "Analysis outputs written")
    return {
        "output_dir": out,
        "images": images,
        "field_summary": field_summary_public,
        "slide_summary": slide_summary,
        "all_chords": all_chords,
        "processing_log": log_df_public,
        "preview_paths": preview_paths,
    }
