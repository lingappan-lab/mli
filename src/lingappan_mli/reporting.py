"""Reporting and export helpers for Lingappan MLI results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .contours import (
    AREA_MEASUREMENT_METHOD,
    airspace_component_connectivity_label,
    area_measurement_method_description,
)

FIELD_EXPORT_COLUMNS = [
    ("filename", "Image"),
    ("slide_id", "Slide"),
    ("field_id", "Field"),
    ("mli_orientation_balanced_mean_um", "MLI, orientation-balanced (µm)"),
    ("mli_pooled_chord_mean_um", "MLI pooled chord mean (µm)"),
    ("mli_chord_count", "MLI chords (n)"),
    ("mli_horizontal_mean_um", "MLI horizontal mean (µm)"),
    ("mli_vertical_mean_um", "MLI vertical mean (µm)"),
    ("mli_direct_horizontal_vertical_mean_ratio", "MLI H/V ratio"),
    ("airspace_fraction", "All airspace fraction"),
    ("non_edge_airspace_fraction", "Non-edge airspace fraction"),
    ("non_edge_airspace_component_area_um2_mask", "Non-edge airspace area (µm²)"),
    ("largest_airspace_component_fraction_of_airspace", "Largest component fraction"),
    ("mean_non_airspace_chord_um", "Non-airspace chord, orientation-balanced (µm)"),
    ("pooled_mean_non_airspace_chord_um", "Non-airspace pooled chord mean (µm)"),
    ("non_airspace_chord_count", "Non-airspace chords (n)"),
    ("calibration_warning", "Calibration warning"),
    ("image_load_warnings", "Image warning"),
]

SLIDE_EXPORT_COLUMNS = [
    ("slide_id", "Slide"),
    ("field_count", "Fields (n)"),
    ("mli_measured_field_count", "MLI measured fields (n)"),
    ("field_balanced_mean_mli_um", "Mean MLI, field-balanced (µm)"),
    ("field_sd_mli_um", "MLI SD across fields (µm)"),
    ("field_sem_mli_um", "MLI SEM (µm)"),
    ("chord_pooled_mean_mli_um", "MLI chord-pooled mean (µm)"),
    ("mean_mli_chords_per_field", "Mean MLI chords/field"),
    ("total_mli_chords", "Total MLI chords"),
    ("mean_airspace_fraction", "All airspace fraction"),
    ("mean_non_edge_airspace_fraction", "Non-edge airspace fraction"),
    ("mean_non_edge_airspace_component_area_um2_mask", "Non-edge airspace area (µm²)"),
    ("mean_largest_airspace_component_fraction_of_airspace", "Largest component fraction"),
    ("non_airspace_measured_field_count", "Non-airspace measured fields (n)"),
    ("field_balanced_mean_non_airspace_chord_um", "Mean non-airspace chord, field-balanced (µm)"),
    ("field_sd_non_airspace_chord_um", "Non-airspace SD across fields (µm)"),
    ("field_sem_non_airspace_chord_um", "Non-airspace SEM (µm)"),
    ("chord_pooled_mean_non_airspace_chord_um", "Non-airspace chord-pooled mean (µm)"),
    ("mean_non_airspace_chords_per_field", "Mean non-airspace chords/field"),
    ("total_non_airspace_chords", "Total non-airspace chords"),
    ("calibration_is_default", "Default calibration?"),
    ("calibration_source", "Calibration source"),
]

FIELD_QC_EXPORT_COLUMNS = [
    ("filename", "Image"),
    ("slide_id", "Slide"),
    ("field_id", "Field"),
    ("mli_chord_count_raw", "MLI raw candidates"),
    ("mli_chord_count_min_length_excluded", "MLI min-length excluded"),
    ("mli_chord_count_edge_excluded", "MLI edge excluded"),
    ("mli_chord_count_accepted", "MLI accepted"),
    ("mli_chord_fraction_min_length_excluded", "MLI min-length excluded fraction"),
    ("mli_chord_fraction_edge_excluded", "MLI edge excluded fraction"),
    ("mli_indirect_equivalent_orientation_balanced_mean_um", "MLI indirect-equivalent (µm)"),
    ("mli_total_airspace_line_length_um_accepted", "MLI accepted line length (µm)"),
    ("mli_airspace_boundary_intersection_count_accepted", "MLI accepted boundary intersections"),
    ("non_airspace_chord_count_raw", "Non-airspace raw candidates"),
    ("non_airspace_chord_count_min_length_excluded", "Non-airspace min-length excluded"),
    ("non_airspace_chord_count_edge_excluded", "Non-airspace edge excluded"),
    ("non_airspace_chord_count_accepted", "Non-airspace accepted"),
    ("non_airspace_chord_fraction_min_length_excluded", "Non-airspace min-length excluded fraction"),
    ("non_airspace_chord_fraction_edge_excluded", "Non-airspace edge excluded fraction"),
    ("airspace_component_count_all", "Airspace components"),
    ("airspace_component_count_edge_excluded", "Edge components excluded"),
    ("largest_airspace_component_area_um2", "Largest component area (µm²)"),
    ("largest_airspace_component_touches_edge", "Largest component touches edge?"),
    ("calibration_warning", "Calibration warning"),
    ("image_load_warnings", "Image warning"),
]

CHORD_EXPORT_COLUMNS = [
    ("filename", "Image"),
    ("slide_id", "Slide"),
    ("field_id", "Field"),
    ("measure_type", "Measurement"),
    ("orientation", "Orientation"),
    ("line_id", "Line"),
    ("line_position_px", "Line position (px)"),
    ("length_um", "Length (µm)"),
    ("length_px", "Length (px)"),
    ("edge_touching", "Edge-touching?"),
]

COMPACT_COLUMN_MEANINGS = {
    "filename": "Original input image filename.",
    "slide_id": "Slide or specimen identifier parsed from the filename.",
    "field_id": "Field or ROI identifier parsed from the filename.",
    "mli_orientation_balanced_mean_um": (
        "Field-level MLI: the orientation-balanced mean accepted airspace chord length."
    ),
    "mli_pooled_chord_mean_um": (
        "Field-level MLI mean after pooling all accepted airspace chords across generated line orientations."
    ),
    "mli_chord_count": "Number of accepted airspace chords retained for the field-level MLI estimate.",
    "mli_horizontal_mean_um": "Mean accepted airspace chord length on horizontal test lines.",
    "mli_vertical_mean_um": "Mean accepted airspace chord length on vertical test lines.",
    "mli_direct_horizontal_vertical_mean_ratio": (
        "Horizontal MLI chord mean divided by vertical MLI chord mean; a directional anisotropy diagnostic."
    ),
    "airspace_fraction": "All segmented airspace fraction before edge exclusion.",
    "non_edge_airspace_fraction": (
        "Fraction of analyzable pixels occupied by fully internal/non-edge airspace components."
    ),
    "non_edge_airspace_component_area_um2_mask": (
        "Fully internal connected airspace-component area from filled mask pixels after edge exclusion."
    ),
    "largest_airspace_component_fraction_of_airspace": (
        "Fraction of all segmented airspace pixels contained in the largest connected component."
    ),
    "mean_non_airspace_chord_um": (
        "Field-level orientation-balanced mean accepted non-airspace chord length; not wall thickness."
    ),
    "pooled_mean_non_airspace_chord_um": (
        "Field-level mean after pooling accepted non-airspace chords across generated line orientations."
    ),
    "non_airspace_chord_count": "Number of accepted non-airspace chords retained for the field.",
    "calibration_warning": "Calibration warning recorded for the run or field, usually default calibration use.",
    "image_load_warnings": "Image-loading and resolution/calibration warnings such as non-uint8 scaling, multi-page first-frame use, or embedded-resolution mismatch.",
    "field_count": "Number of analyzed fields included in the slide-level summary.",
    "mli_measured_field_count": "Number of analyzed fields with a valid field-level MLI measurement.",
    "field_balanced_mean_mli_um": (
        "Slide-level mean of field MLI values with each field weighted equally."
    ),
    "field_sd_mli_um": "Sample standard deviation of field-level MLI values across fields.",
    "field_sem_mli_um": "Standard error of the field-balanced MLI mean across fields.",
    "chord_pooled_mean_mli_um": (
        "Slide-level mean after pooling accepted MLI chords across fields; fields with more chords weigh more."
    ),
    "mean_mli_chords_per_field": "Mean accepted MLI chord count per analyzed field.",
    "total_mli_chords": "Total accepted MLI chord count across fields.",
    "mean_airspace_fraction": "Mean all-airspace fraction across fields before edge exclusion.",
    "mean_non_edge_airspace_fraction": (
        "Mean fraction occupied by fully internal/non-edge airspace components across fields."
    ),
    "mean_non_edge_airspace_component_area_um2_mask": (
        "Mean fully internal connected airspace-component mask area across fields."
    ),
    "mean_largest_airspace_component_fraction_of_airspace": (
        "Mean largest-component fraction across fields, used as a segmentation QC diagnostic."
    ),
    "non_airspace_measured_field_count": (
        "Number of analyzed fields with a valid preferred non-airspace chord measurement."
    ),
    "field_balanced_mean_non_airspace_chord_um": (
        "Slide-level mean of field non-airspace chord values with each field weighted equally; not wall thickness."
    ),
    "field_sd_non_airspace_chord_um": (
        "Sample standard deviation of field-level non-airspace chord values across fields."
    ),
    "field_sem_non_airspace_chord_um": (
        "Standard error of the field-balanced non-airspace chord mean across fields."
    ),
    "chord_pooled_mean_non_airspace_chord_um": (
        "Slide-level mean after pooling accepted non-airspace chords across fields; not wall thickness."
    ),
    "mean_non_airspace_chords_per_field": "Mean accepted non-airspace chord count per analyzed field.",
    "total_non_airspace_chords": "Total accepted non-airspace chord count across fields.",
    "calibration_is_default": "Whether the built-in default pixel-size calibration was used.",
    "calibration_source": "Recorded source or assumption for the pixel-size calibration.",
    "mli_chord_count_raw": "Raw airspace chord candidates before minimum-length and edge-touching filters.",
    "mli_chord_count_min_length_excluded": "Raw airspace chord candidates removed by the minimum-length filter.",
    "mli_chord_count_edge_excluded": "Airspace chord candidates removed by edge-touching exclusion.",
    "mli_chord_count_accepted": "Final accepted airspace chord count after all chord filters.",
    "mli_chord_fraction_min_length_excluded": (
        "Fraction of raw airspace chord candidates removed by the minimum-length filter."
    ),
    "mli_chord_fraction_edge_excluded": (
        "Fraction of include-edge airspace chord candidates removed by edge-touching exclusion."
    ),
    "mli_indirect_equivalent_orientation_balanced_mean_um": (
        "Orientation-balanced indirect-equivalent MLI from accepted line length and paired boundary counts."
    ),
    "mli_total_airspace_line_length_um_accepted": (
        "Total accepted airspace line length sampled by the MLI test lines."
    ),
    "mli_airspace_boundary_intersection_count_accepted": (
        "Accepted in-line airspace/non-airspace boundary intersection count for MLI consistency checks."
    ),
    "non_airspace_chord_count_raw": "Raw non-airspace chord candidates before minimum-length and edge-touching filters.",
    "non_airspace_chord_count_min_length_excluded": (
        "Raw non-airspace chord candidates removed by the minimum-length filter."
    ),
    "non_airspace_chord_count_edge_excluded": "Non-airspace chord candidates removed by edge-touching exclusion.",
    "non_airspace_chord_count_accepted": "Final accepted non-airspace chord count after all chord filters.",
    "non_airspace_chord_fraction_min_length_excluded": (
        "Fraction of raw non-airspace chord candidates removed by the minimum-length filter."
    ),
    "non_airspace_chord_fraction_edge_excluded": (
        "Fraction of include-edge non-airspace chord candidates removed by edge-touching exclusion."
    ),
    "airspace_component_count_all": "Connected airspace-component count before edge exclusion.",
    "airspace_component_count_edge_excluded": "Connected airspace-component count removed by edge exclusion.",
    "largest_airspace_component_area_um2": "Area of the largest connected airspace component.",
    "largest_airspace_component_touches_edge": "Whether the largest connected airspace component touches the image edge.",
    "measure_type": "Chord measurement class: MLI airspace or non-airspace.",
    "orientation": "Test-line orientation used to sample the chord.",
    "line_id": "Identifier of the generated test line that produced the chord.",
    "line_position_px": "Pixel coordinate of the generated test line.",
    "length_um": "Chord length after pixel-size calibration.",
    "length_px": "Chord length in pixels.",
    "edge_touching": "Whether the chord touches the image boundary.",
}


def _compact_export_frame(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.DataFrame:
    """Return user-facing export columns with readable headers."""
    present = [(source, label) for source, label in columns if source in frame.columns]
    if not present:
        return pd.DataFrame()
    out = frame[[source for source, _ in present]].copy()
    out = out.rename(columns={source: label for source, label in present})
    return out


def _compact_run_log(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame(columns=["Item", "Status", "Message", "Included in summaries?", "Exclusion reason"])
    columns = [
        ("filename", "Item"),
        ("status", "Status"),
        ("message", "Message"),
        ("field_included_in_summary", "Included in summaries?"),
        ("field_exclusion_reason", "Exclusion reason"),
    ]
    return _compact_export_frame(log_df, columns)


def _run_settings_sheet(params, *, pixel_area_um2: float, calibration_is_default: bool) -> pd.DataFrame:
    connectivity_label = airspace_component_connectivity_label(params.airspace_component_connectivity)
    rows = [
        ("Input", "Input image type", params.input_image_type),
        ("Input", "Field selection method", params.field_selection_method),
        ("Input", "Field exclusion criteria", params.field_exclusion_criteria),
        ("Calibration", "Pixel width (µm/px)", params.pixel_width_um),
        ("Calibration", "Pixel height (µm/px)", params.pixel_height_um),
        ("Calibration", "Pixel area (µm²/px)", pixel_area_um2),
        ("Calibration", "Calibration source", params.calibration_source),
        ("Calibration", "Default calibration?", calibration_is_default),
        ("Segmentation", "Threshold method", params.threshold_method),
        ("Segmentation", "Airspace polarity", "bright" if params.airspace_bright else "dark"),
        ("Airspace components", "Connectivity", connectivity_label),
        ("Grid", "Grid strategy", params.grid_strategy),
        ("Grid", "Lines per orientation", params.num_lines),
        ("Grid", "Line spacing (µm)", params.line_spacing_um),
        ("Grid", "Orientation", params.orientation),
        ("Grid", "Random grid offset?", params.grid_random_offset),
        ("Grid", "Random seed", params.grid_random_seed if params.grid_random_seed is not None else "none"),
        ("Measurement", "Measure non-airspace chords?", params.measure_non_airspace),
        ("Filtering", "Exclude edge-touching chords/components?", params.exclude_edge_touching),
        ("Filtering", "Minimum chord length (µm)", params.min_chord_um),
        ("Output", "Area measurement method", AREA_MEASUREMENT_METHOD),
    ]
    return pd.DataFrame(rows, columns=["Group", "Setting", "Value"])


def _compact_column_units(source: str, label: str) -> str:
    label_lower = label.lower()
    source_lower = source.lower()
    if "µm²" in label or "um2" in source_lower:
        return "µm²"
    if "µm" in label or source_lower.endswith("_um") or "_um_" in source_lower:
        return "µm"
    if "(px)" in label_lower or source_lower.endswith("_px") or "_px_" in source_lower:
        return "px"
    if "fraction" in label_lower or "ratio" in label_lower:
        return "fraction"
    if label.endswith("?") or source_lower in {
        "calibration_is_default",
        "largest_airspace_component_touches_edge",
        "edge_touching",
    }:
        return "boolean"
    if (
        "(n)" in label_lower
        or "count" in source_lower
        or "chords" in label_lower
        or "boundary_intersection" in source_lower
    ):
        return "count"
    return ""


def _column_dictionary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sheet, group, columns in (
        ("Fields", "Primary field results", FIELD_EXPORT_COLUMNS),
        ("Slides", "Primary slide results", SLIDE_EXPORT_COLUMNS),
        ("Field QC", "Secondary field QC", FIELD_QC_EXPORT_COLUMNS),
        ("all_chords.csv", "Chord data", CHORD_EXPORT_COLUMNS),
    ):
        for source, label in columns:
            rows.append(
                {
                    "Sheet": sheet,
                    "Group": group,
                    "Column": label,
                    "Source column": source,
                    "Units": _compact_column_units(source, label),
                    "Meaning": COMPACT_COLUMN_MEANINGS.get(source, f"Compact export value for `{source}`."),
                }
            )
    return pd.DataFrame(rows)


def _write_results_readme(
    output_dir: Path,
    params,
    *,
    pixel_area_um2: float,
    calibration_warning: str,
    export_warnings: list[tuple[str, str]] | None = None,
) -> None:
    connectivity_label = airspace_component_connectivity_label(params.airspace_component_connectivity)
    method_description = area_measurement_method_description(params.airspace_component_connectivity)
    calibration_warning = calibration_warning or "none"
    export_warning_text = "\n"
    if export_warnings:
        warning_rows = "\n".join(f"- `{item}`: {message}" for item, message in export_warnings)
        export_warning_text = (
            "\n## Export warnings\n\n"
            "One or more workbook exports could not be written. "
            "CSV/audit files that were successfully created remain available, "
            "and the processing log records these warnings.\n\n"
            f"{warning_rows}\n\n"
        )
    text = f"""# Lingappan MLI Analyzer Results

This folder contains outputs from the semi-automated MLI analysis.

## Method summary

- Input images are assumed to be pre-cropped lung fields/ROIs (`{params.input_image_type}`).
- Whole-slide field selection and any protocol-driven field exclusions must occur upstream; this analyzer records field-selection metadata but does not choose ROIs.
- Field-selection method: `{params.field_selection_method}`.
- Field-exclusion criteria recorded for this run: `{params.field_exclusion_criteria}`.
- Image loading records the original file format, Pillow mode, NumPy dtype, analyzed frame/page index, page count when available, embedded resolution metadata when available, and any load warnings. Multi-page TIFFs use frame/page 0. Non-uint8 numeric image data is min-max scaled to uint8 before thresholding, with the finite scaling range recorded and logged.
- Images are converted to 8-bit grayscale.
- Calibration: `{params.pixel_width_um:g}` × `{params.pixel_height_um:g}` µm/px; pixel area `{pixel_area_um2:g}` µm²/px; source `{params.calibration_source}`. Default-calibration warning: `{calibration_warning}`. Plausible embedded image resolution metadata is compared with the analysis calibration for warnings, but never automatically overrides the supplied calibration.
- Field summaries record pixel dimensions plus physical field width, height, and area in micrometers. Batch-level warnings report mixed pixel dimensions or >2% physical field-size variation across analyzed fields.
- Airspace is segmented using `{params.threshold_method}` thresholding.
- Area measurement method: `{AREA_MEASUREMENT_METHOD}` (`{connectivity_label}` components). {method_description}
- Systematic horizontal/vertical test lines are generated with strategy `{params.grid_strategy}`.
- Systematic-random grid phase offsets are {'enabled' if params.grid_random_offset else 'disabled'}; run seed: `{params.grid_random_seed if params.grid_random_seed is not None else 'none'}`. Per-field summaries record the derived field seed and horizontal/vertical offsets in pixels and micrometers.
- MLI is reported as the mean length of accepted continuous airspace segments along the test lines. `mli_orientation_balanced_direct_mean_um` and `mli_pooled_chord_mean_um` distinguish the orientation-balanced mean from the pooled chord mean.
- Slide summaries make weighting explicit: field-balanced means/SD/SEM use fields as the experimental unit, while `chord_pooled_mean_*` columns pool accepted chords across fields and therefore weight fields by chord count. Mean accepted chords per field is reported for MLI and, when measured, non-airspace chords.
- Orientation anisotropy diagnostics report horizontal/vertical direct mean ratios and signed horizontal-minus-vertical deltas for MLI and, when measured, non-airspace chords.
- Filter-sensitivity QC columns report raw, include-edge, minimum-length-excluded, edge-excluded, and accepted chord counts/means/fractions for MLI and, when measured, non-airspace chords.
- Indirect-equivalent consistency metrics report total airspace line length and airspace-boundary intersection counts. `mli_indirect_equivalent_*mean_um` is populated only when accepted chords have complete paired in-line boundaries, so `2 × accepted airspace line length / accepted boundary intersections` is equivalent to the direct chord mean.
- Non-airspace chords are {'measured' if params.measure_non_airspace else 'not measured'}. These are line-intercept chord statistics through non-airspace pixels, not septal wall-thickness measurements.
- Edge-touching chords and connected airspace components are {'excluded' if params.exclude_edge_touching else 'included'} for final filtered metrics.
- `airspace_fraction` reports all segmented analysis-mask airspace before edge exclusion. `non_edge_airspace_fraction` reports the fraction occupied by fully internal/non-edge airspace components using the full-image denominator.
- Preferred `non_edge_airspace_component_area_*` columns report fully internal filled connected-component pixel-count area; this edge-censored area is not an unbiased estimate of total airspace area, and connected components should not be interpreted as individual alveoli. Largest-component area/fraction columns are QC diagnostics for dominant segmented airspace objects.

## Key files

- `lingappan_mli_results.xlsx`: compact, human-readable workbook with `Slides`, `Fields`, `Field QC`, `Run settings`, `Run log`, `Column dictionary`, and `Export notes` sheets.
- `all_chords.csv`: compact, human-readable chord table for all images.
- `audit/lingappan_mli_audit.xlsx`: full machine-readable field, slide, and processing-log audit tables.
- `audit/all_chords_audit.csv`: complete machine-readable chord-level table with coordinates and all audit columns.
- `parameters.json`: full analysis settings plus calibration metadata and area-measurement metadata (`area_measurement_method={AREA_MEASUREMENT_METHOD}`, `area_measurement_connectivity={connectivity_label}`).
- Per-image folders:
  - `01_preprocessing/`: grayscale and binary analysis-mask PNGs.
  - `02_contours/`: all-component, final filtered, and filled connected-component contour QC PNGs.
  - `03_mli/`: MLI test-line/chord PNG outputs and overlay.
  - `04_non_airspace/`: non-airspace chord PNG outputs and overlay when enabled.
  - `05_data/`: compact per-image `chords.csv` plus `chords_audit.csv` with complete machine-readable chord metadata; stored here only to avoid duplicating data files at the image-folder root.
  - `06_qc/`: first-class QC overlays, including `mask_overlay.png` and rejected-edge chord overlays for MLI and, when enabled, non-airspace chords.
  - `qc_panel.png`: combined QC panel.
{export_warning_text}## Column reference

The primary workbook sheets (`Slides`, `Fields`, and `Field QC`) are intentionally compact and use human-readable headers. Detailed reproducibility and audit columns are moved to `audit/lingappan_mli_audit.xlsx`, `parameters.json`, and the `*_audit.csv` files.

### Audit workbook sheet: `field_summary` in `audit/lingappan_mli_audit.xlsx`

| Column | Meaning |
| --- | --- |
| `filename` | Original image filename. |
| `slide_id` | Slide/specimen ID parsed from the filename prefix. |
| `field_id` | Field/ROI ID parsed from the filename suffix. |
| `input_image_type` | Fixed audit value `pre_cropped_lung_roi`; inputs are assumed to be cropped ROIs, not whole-slide images. |
| `field_selection_stage`, `field_selected_by_app` | Records that field/ROI selection happened upstream and not inside this app. |
| `field_selection_method`, `field_selection_notes`, `field_exclusion_criteria` | Run-level metadata describing upstream field selection and field-exclusion criteria supplied by the user. |
| `field_included_in_summary`, `field_exclusion_flag`, `field_exclusion_reason` | Whether this image contributed to field/slide summaries and why not; successfully analyzed field rows are included and unflagged. |
| `edge_exclusion_applied`, `short_chord_exclusion_applied` | Boolean audit flags for edge-touching and minimum-length chord filtering. |
| `airspace_component_connectivity`, `area_measurement_method`, `area_measurement_connectivity`, `area_column_semantics`, `area_measurement_method_description` | Area-method metadata recording the configured 4- or 8-connectivity and that area columns are filled connected-component pixel counts, not geometric contour integration. |
| `calibration_source`, `calibration_is_default`, `calibration_warning` | Source/assumption for the pixel-size calibration; default 0.57 × 0.57 µm/px calibration is explicitly flagged. |
| `width_px`, `height_px` | Image dimensions in pixels after loading the analyzed frame/page. |
| `image_original_format`, `image_original_mode`, `image_original_dtype`, `image_output_dtype` | Image-load audit metadata recording the source format, original Pillow mode, original NumPy dtype, and analysis array dtype. |
| `image_frame_index`, `image_page_count` | Frame/page index analyzed (0) and total page/frame count when Pillow reports it; multi-page TIFFs use only the first page. |
| `image_scaling_applied`, `image_scaling_min`, `image_scaling_max` | Whether non-uint8 numeric data was min-max scaled to uint8 before thresholding and the finite intensity range used. |
| `image_embedded_resolution_x`, `image_embedded_resolution_y`, `image_embedded_resolution_unit`, `image_embedded_resolution_source` | Embedded TIFF/DPI resolution values and source metadata when available. |
| `image_embedded_pixel_width_um`, `image_embedded_pixel_height_um` | Physical pixel size implied by embedded resolution metadata when the unit is convertible. |
| `image_load_warnings` | Per-image load warnings, including non-uint8 scaling, multi-page TIFF first-page use, and embedded-resolution/calibration mismatches. |
| `field_width_um`, `field_height_um`, `field_area_um2` | Physical field dimensions computed from image dimensions and calibration. |
| `pixel_width_um`, `pixel_height_um`, `pixel_area_um2` | Calibration used to convert pixels to micrometers and square micrometers. |
| `threshold_method`, `threshold_value` | Segmentation method and threshold value used for airspace detection. |
| `airspace_fraction` | All segmented airspace fraction before edge exclusion over the full image. |
| `non_edge_airspace_fraction` | Fraction of the full image occupied by fully internal/non-edge airspace components after removing edge-touching components. |
| `grid_strategy`, `num_lines_requested`, `line_spacing_um_requested`, `orientation` | Requested test-line setup. |
| `grid_random_offset`, `grid_random_seed`, `grid_random_field_seed` | Whether systematic-random grid phase offsets were enabled, the recorded run seed, and the per-field seed derived from it. |
| `grid_horizontal_offset_px`, `grid_vertical_offset_px`, `grid_horizontal_offset_um`, `grid_vertical_offset_um` | Applied phase shift from the deterministic grid for horizontal and vertical line sets; blank for orientations not generated. |
| `horizontal_lines`, `vertical_lines` | Number of generated test lines by orientation. |
| `exclude_edge_touching`, `min_chord_um` | Filtering settings used for incomplete edge chords and short chords. |
| `airspace_component_count_all` | Number of connected airspace components before edge exclusion; components are segmentation objects, not individual alveoli. |
| `airspace_component_count_final` | Number of connected airspace components retained for final filtered area metrics. |
| `airspace_component_count_edge_excluded` | Number of edge-touching connected airspace components excluded from final metrics. |
| `largest_airspace_component_area_pixels`, `largest_airspace_component_area_um2` | Area of the largest connected airspace component in pixels and µm², reported as a QC diagnostic for dominant segmented objects. |
| `largest_airspace_component_fraction_of_airspace`, `largest_airspace_component_touches_edge` | Fraction of all segmented airspace pixels contained in the largest component, and whether that component touches the field edge. |
| `total_airspace_area_pixels_mask`, `total_airspace_area_um2_mask` | Total binary-mask airspace area before edge exclusion, in pixels and µm². |
| `final_airspace_area_pixels_mask`, `final_airspace_area_um2_mask` | Binary-mask area columns after configured edge handling; when edge exclusion is enabled, these equal the non-edge component-area columns. |
| `non_edge_airspace_component_area_pixels_mask`, `non_edge_airspace_component_area_um2_mask` | Fully internal/non-edge connected-component mask area in pixels and µm²; this edge-censored area is not an unbiased estimate of total airspace area. |
| `mli_chord_count`, `non_airspace_chord_count` | Number of accepted airspace chords for MLI and non-airspace chords when measured. |
| `mli_chord_count_raw`, `non_airspace_chord_count_raw` | Raw phase-run counts before edge and minimum-length chord filters. |
| `mli_chord_count_include_edge`, `non_airspace_chord_count_include_edge` | Candidate chord counts after minimum-length filtering with edge-touching chords retained. |
| `mli_chord_count_min_length_excluded`, `non_airspace_chord_count_min_length_excluded` | Raw candidate chords removed by the configured minimum-length filter before edge filtering. |
| `mli_chord_count_edge_excluded`, `non_airspace_chord_count_edge_excluded` | Candidate chord counts removed by the configured edge-touching exclusion. |
| `mli_chord_fraction_min_length_excluded`, `non_airspace_chord_fraction_min_length_excluded` | Fraction of raw candidates removed by the minimum-length filter. |
| `mli_chord_fraction_edge_excluded`, `non_airspace_chord_fraction_edge_excluded` | Fraction of include-edge candidates removed by edge-touching exclusion. |
| `mli_chord_count_accepted`, `non_airspace_chord_count_accepted` | Final chord counts after edge and minimum-length filters. |
| `mli_horizontal_count`, `mli_vertical_count`, `non_airspace_horizontal_count`, `non_airspace_vertical_count` | Accepted chord counts by orientation. |
| `*_horizontal_count_raw`, `*_vertical_count_raw`, `*_horizontal_count_include_edge`, `*_vertical_count_include_edge`, `*_horizontal_count_min_length_excluded`, `*_vertical_count_min_length_excluded`, `*_horizontal_count_edge_excluded`, `*_vertical_count_edge_excluded`, `*_horizontal_count_accepted`, `*_vertical_count_accepted` | Orientation-specific raw, include-edge, minimum-length-excluded, edge-excluded, and accepted chord counts for MLI/non-airspace measurements. |
| `mli_mean_um`, `non_airspace_mean_um` | Pooled direct mean chord length across all accepted chords. |
| `mli_direct_mean_um`, `non_airspace_direct_mean_um` | Explicit aliases for the pooled direct chord means. |
| `mli_mean_um_raw`, `non_airspace_mean_um_raw`; `mli_mean_um_include_edge`, `non_airspace_mean_um_include_edge`; `mli_mean_um_edge_excluded`, `non_airspace_mean_um_edge_excluded`; `mli_mean_um_accepted`, `non_airspace_mean_um_accepted` | Pooled raw, include-edge, edge-excluded, and accepted chord means for filter sensitivity. |
| `mli_pooled_chord_mean_um`, `non_airspace_pooled_chord_mean_um` | Clear aliases for the pooled chord means. |
| `mean_non_airspace_chord_um`, `orientation_balanced_mean_non_airspace_chord_um`, `pooled_mean_non_airspace_chord_um` | Preferred aliases for the non-airspace chord statistic. The first two report the orientation-balanced direct mean; the pooled alias reports the pooled accepted non-airspace chord mean. These are not wall-thickness measurements. |
| `mli_orientation_balanced_mean_um`, `non_airspace_orientation_balanced_mean_um` | Direct mean of the horizontal and vertical means when both orientations are present. |
| `mli_orientation_balanced_mean_um_raw`, `non_airspace_orientation_balanced_mean_um_raw`; `mli_orientation_balanced_mean_um_include_edge`, `non_airspace_orientation_balanced_mean_um_include_edge`; `mli_orientation_balanced_mean_um_edge_excluded`, `non_airspace_orientation_balanced_mean_um_edge_excluded`; `mli_orientation_balanced_mean_um_accepted`, `non_airspace_orientation_balanced_mean_um_accepted` | Orientation-balanced raw, include-edge, edge-excluded, and accepted chord means for filter sensitivity. |
| `mli_direct_orientation_balanced_mean_um`, `non_airspace_direct_orientation_balanced_mean_um` | Explicit aliases for the orientation-balanced direct chord means. |
| `mli_orientation_balanced_direct_mean_um`, `non_airspace_orientation_balanced_direct_mean_um` | Clear aliases distinguishing the orientation-balanced direct means from pooled chord means. |
| `mli_horizontal_mean_um`, `mli_vertical_mean_um`, `non_airspace_horizontal_mean_um`, `non_airspace_vertical_mean_um` | Direct mean chord length by orientation. |
| `mli_direct_horizontal_mean_um`, `mli_direct_vertical_mean_um`, `non_airspace_direct_horizontal_mean_um`, `non_airspace_direct_vertical_mean_um` | Explicit aliases for the orientation-specific direct chord means. |
| `mli_direct_horizontal_vertical_mean_ratio`, `non_airspace_direct_horizontal_vertical_mean_ratio` | Orientation anisotropy ratio: horizontal direct mean divided by vertical direct mean; blank unless both orientations have accepted chords. |
| `mli_direct_horizontal_vertical_mean_delta_um`, `non_airspace_direct_horizontal_vertical_mean_delta_um` | Orientation anisotropy delta: horizontal direct mean minus vertical direct mean, in µm; blank unless both orientations have accepted chords. |
| `mli_total_airspace_line_length_um_raw`, `mli_total_airspace_line_length_um_accepted` | Sum of raw and accepted MLI airspace chord lengths sampled by the test lines. |
| `mli_horizontal_airspace_line_length_um_raw`, `mli_vertical_airspace_line_length_um_raw` | Raw airspace line length by orientation before chord filters. |
| `mli_horizontal_airspace_line_length_um_accepted`, `mli_vertical_airspace_line_length_um_accepted` | Accepted airspace line length by orientation after chord filters. |
| `mli_airspace_boundary_intersection_count_raw`, `mli_airspace_boundary_intersection_count_accepted` | Number of in-line airspace/non-airspace boundary crossings before and after chord filters. Edge contacts are not counted as observed boundaries. |
| `mli_horizontal_airspace_boundary_intersection_count_raw`, `mli_vertical_airspace_boundary_intersection_count_raw` | Raw airspace-boundary crossings by orientation. |
| `mli_horizontal_airspace_boundary_intersection_count_accepted`, `mli_vertical_airspace_boundary_intersection_count_accepted` | Accepted airspace-boundary crossings by orientation. |
| `mli_indirect_equivalent_mean_um` | Pooled indirect-equivalent MLI over accepted chords: `2 × accepted airspace line length / accepted boundary intersections`, blank when accepted chords do not have paired observed boundaries. |
| `mli_indirect_equivalent_orientation_balanced_mean_um` | Orientation-balanced indirect-equivalent MLI, calculated from orientation-specific indirect-equivalent means when applicable. |
| `mli_indirect_equivalent_horizontal_mean_um`, `mli_indirect_equivalent_vertical_mean_um` | Orientation-specific indirect-equivalent MLI values when accepted chords in that orientation have paired observed boundaries. |
| `mli_median_um`, `non_airspace_median_um` | Median accepted chord length. |
| `mli_sd_um`, `non_airspace_sd_um` | Sample standard deviation of accepted chord lengths. |

### Audit workbook sheet: `slide_summary` in `audit/lingappan_mli_audit.xlsx`

| Column | Meaning |
| --- | --- |
| `slide_id` | Slide/specimen ID used for grouping fields. |
| `field_count` | Number of fields included for the slide; fields are the experimental unit for field-balanced means/SEM. |
| `airspace_component_connectivity`, `area_measurement_connectivity` | Run-level component-connectivity metadata carried into slide summaries for auditability. |
| `calibration_source`, `calibration_is_default`, `pixel_area_um2` | Calibration metadata carried into slide summaries for auditability. |
| `mean_field_width_um`, `mean_field_height_um`, `mean_field_area_um2` | Mean analyzed field dimensions for the slide. |
| `mean_airspace_fraction` | Mean all-airspace fraction before edge exclusion. |
| `mean_non_edge_airspace_fraction` | Mean non-edge airspace fraction across fields, using each field's full-image denominator. |
| `mean_total_airspace_area_um2_mask`, `mean_final_airspace_area_um2_mask` | Mean mask-based airspace area before and after configured edge handling. |
| `mean_non_edge_airspace_component_area_um2_mask` | Mean fully internal/non-edge connected-component mask area; this edge-censored mean is not an unbiased total airspace estimate. |
| `mean_largest_airspace_component_area_um2`, `mean_largest_airspace_component_fraction_of_airspace` | Slide-level means of largest-component QC diagnostics. |
| `mean_mli_um`, `median_mli_um`, `sd_mli_um` | Field-balanced mean, median, and field SD of field direct-chord MLI; each field has equal weight. |
| `field_balanced_mean_mli_um`, `field_sd_mli_um`, `field_sem_mli_um`, `sem_mli_um` | Explicit field-balanced MLI mean plus field SD/SEM across field-level MLI values. Fields, not chords, are the experimental units for these summaries. |
| `chord_pooled_mean_mli_um`, `mean_mli_chords_per_field` | Supplementary MLI mean pooling all accepted chords across fields (therefore weighting fields by chord count), and the mean accepted MLI chord count per field. |
| `mean_mli_direct_um`, `mean_mli_orientation_balanced_direct_um`, `mean_mli_pooled_chord_um`, `mean_mli_indirect_equivalent_um` | Optional field-balanced slide-level means of explicit direct-chord, field pooled-chord, and indirect-equivalent field MLI columns. `mean_mli_pooled_chord_um` averages per-field pooled means and is not the chord-pooled slide mean. |
| `mean_mli_raw_um`, `mean_mli_include_edge_um`, `mean_mli_edge_excluded_um`, `mean_mli_accepted_um`, `mean_mli_chord_fraction_min_length_excluded`, `mean_mli_chord_fraction_edge_excluded` | Optional slide-level means of field MLI filter-sensitivity columns. |
| `mean_mli_horizontal_vertical_mean_ratio`, `mean_mli_horizontal_vertical_mean_delta_um` | Optional slide-level means of field MLI horizontal/vertical anisotropy diagnostics. |
| `total_mli_chords` | Total accepted MLI chord count across fields. |
| `total_mli_chords_raw`, `total_mli_chords_include_edge`, `total_mli_chords_min_length_excluded`, `total_mli_chords_edge_excluded`, `total_mli_chords_accepted` | Optional raw, include-edge, minimum-length-excluded, edge-excluded, and accepted MLI chord totals across fields. |
| `total_mli_airspace_line_length_um_raw`, `total_mli_airspace_line_length_um_accepted` | Optional summed raw and accepted MLI airspace line length across fields. |
| `total_mli_airspace_boundary_intersections_raw`, `total_mli_airspace_boundary_intersections_accepted` | Optional summed raw and accepted airspace-boundary intersection counts across fields. |
| `mean_non_airspace_chord_um`, `median_non_airspace_chord_um`, `sd_non_airspace_chord_um`, `mean_orientation_balanced_non_airspace_chord_um`, `mean_pooled_non_airspace_chord_um`, `total_non_airspace_chords` | Preferred field-balanced slide-level aliases for non-airspace chord statistics and accepted chord totals; these are line-intercept chord summaries, not wall-thickness measurements. `mean_pooled_non_airspace_chord_um` averages per-field pooled non-airspace means and is not the chord-pooled slide mean. |
| `field_balanced_mean_non_airspace_chord_um`, `field_sd_non_airspace_chord_um`, `field_sem_non_airspace_chord_um`, `sem_non_airspace_chord_um`, `chord_pooled_mean_non_airspace_chord_um`, `mean_non_airspace_chords_per_field` | Preferred explicit non-airspace field-balanced mean, field SD/SEM, supplementary chord-pooled mean, and mean accepted non-airspace chord count per field. |
| `mean_non_airspace_direct_um`, `mean_non_airspace_pooled_chord_um` | Optional field-balanced slide-level means of explicit orientation-balanced and field pooled-chord non-airspace aliases. |
| `mean_non_airspace_raw_um`, `mean_non_airspace_include_edge_um`, `mean_non_airspace_edge_excluded_um`, `mean_non_airspace_accepted_um`, `mean_non_airspace_chord_fraction_min_length_excluded`, `mean_non_airspace_chord_fraction_edge_excluded` | Optional slide-level means of field non-airspace chord filter-sensitivity columns. |
| `mean_non_airspace_horizontal_vertical_mean_ratio`, `mean_non_airspace_horizontal_vertical_mean_delta_um` | Optional slide-level means of field non-airspace chord horizontal/vertical anisotropy diagnostics. |
| `total_non_airspace_chords_raw`, `total_non_airspace_chords_include_edge`, `total_non_airspace_chords_min_length_excluded`, `total_non_airspace_chords_edge_excluded`, `total_non_airspace_chords_accepted` | Optional raw, include-edge, minimum-length-excluded, edge-excluded, and accepted non-airspace chord totals across fields. |

### Workbook sheet: `processing_log`

| Column | Meaning |
| --- | --- |
| `filename` | Image or export item being reported. |
| `status` | `ok`, `warning`, or `error`. |
| `message` | Warning/error details, empty for successful image processing. |
| `traceback` | Python traceback for failed image processing, when available. |
| `input_image_type`, `field_selection_*`, `field_included_in_summary`, `field_exclusion_*`, `calibration_*`, `pixel_area_um2`, `airspace_component_connectivity`, `area_measurement_*`, `area_column_semantics`, `image_*` | Same audit/calibration/area-method and image-load metadata as `field_summary`; failed image rows are flagged because they did not enter summaries. Run-level warnings such as default calibration use, per-image load/resolution warnings, and `batch_qc` field-size consistency warnings may also appear. |

### CSV: `all_chords.csv` and per-image `05_data/chords.csv`

The primary chord CSVs use compact, human-readable columns: `File`, `Slide`, `Field`, `Measurement`, `Orientation`, `Line`, and `Length (µm)`.

Use `audit/all_chords_audit.csv` and per-image `05_data/chords_audit.csv` for the complete machine-readable table with line positions, chord coordinates, pixel lengths, phase labels, and edge-touching flags.

### Workbook sheet: `export_notes`

| Column | Meaning |
| --- | --- |
| `item` | Export note label. |
| `value` | Export note value or explanation. |
"""
    (output_dir / "README_results.md").write_text(text, encoding="utf-8")


def _excel_export_notes(all_chords: pd.DataFrame, params, *, pixel_area_um2: float, calibration_warning: str) -> pd.DataFrame:
    chord_rows = int(len(all_chords))
    grid_phase_value = (
        f"Systematic-random grid phase offsets enabled; run seed {params.grid_random_seed}. "
        "Per-field seeds and offsets are recorded in the audit workbook field_summary sheet."
        if params.grid_random_offset
        else "Deterministic centered grid phase; no random grid offsets applied."
    )
    notes = [
        {
            "item": "Workbook scope",
            "value": "Slides, Fields, and Field QC are compact, human-readable result sheets. Complete machine-readable metadata is preserved in audit/lingappan_mli_audit.xlsx.",
        },
        {
            "item": "Chord CSV scope",
            "value": "all_chords.csv and per-image 05_data/chords.csv are compact chord exports. Use audit/all_chords_audit.csv and per-image chords_audit.csv for full coordinates and audit columns.",
        },
        {
            "item": "Column groups",
            "value": "The Column dictionary sheet lists how compact summary columns map to source audit columns and groups. Workbook header colors use identity, MLI, non-airspace, airspace, QC/warning, and settings groups.",
        },
        {
            "item": "Input scope",
            "value": "Images are treated as pre-cropped lung ROI fields; whole-slide field selection is not performed by this analyzer.",
        },
        {
            "item": "Image load audit",
            "value": (
                "The audit workbook field_summary sheet and processing_log record original image format/mode/dtype, analyzed "
                "frame/page index, page count when available, non-uint8 scaling min/max, embedded resolution metadata, "
                "and load warnings. Multi-page TIFFs use frame/page 0; non-uint8 numeric images are min-max scaled to "
                "uint8 before thresholding. Plausible embedded resolution metadata is compared with the analysis calibration."
            ),
        },
        {
            "item": "Calibration audit",
            "value": (
                f"Pixel size {params.pixel_width_um:g} × {params.pixel_height_um:g} µm/px "
                f"(pixel area {pixel_area_um2:g} µm²/px); source: {params.calibration_source}. "
                + (calibration_warning or "Default calibration was not used.")
            ),
        },
        {
            "item": "Field selection audit",
            "value": "The audit workbook field_summary sheet and processing_log include field-selection metadata plus field/edge/short-chord exclusion flags.",
        },
        {
            "item": "Grid phase audit",
            "value": grid_phase_value,
        },
        {
            "item": "Chord measurement",
            "value": "The app measures each accepted continuous airspace segment along a one-pixel horizontal or vertical test line and converts its length from pixels to micrometers.",
        },
        {
            "item": "Primary MLI and consistency checks",
            "value": "Primary MLI columns are accepted airspace-chord means; mli_orientation_balanced_direct_mean_um and mli_pooled_chord_mean_um distinguish orientation-balanced and pooled chord means, and mli_indirect_equivalent_* columns provide consistency checks from sampled airspace length and paired boundary intersections when applicable.",
        },
        {
            "item": "Slide-level weighting and experimental unit",
            "value": (
                "In slide_summary, field-balanced mean/SD/SEM columns use fields as the experimental "
                "unit and give each field equal weight. chord_pooled_mean_* columns are supplementary "
                "descriptive summaries that pool accepted chords across fields and therefore weight fields "
                "by chord count; mean_*_chords_per_field columns report the chord-count burden behind "
                "that weighting."
            ),
        },
        {
            "item": "Orientation anisotropy diagnostics",
            "value": "The audit workbook field_summary sheet includes horizontal/vertical direct mean ratios and horizontal-minus-vertical deltas for MLI and non-airspace chords when both orientations have accepted chords.",
        },
        {
            "item": "Non-airspace chord caveat",
            "value": "Columns such as mean_non_airspace_chord_um, orientation_balanced_mean_non_airspace_chord_um, and pooled_mean_non_airspace_chord_um describe non-airspace chord statistics. They are not septal wall-thickness measurements.",
        },
        {
            "item": "Chord-filter sensitivity",
            "value": "The audit workbook field_summary sheet reports raw, include-edge, minimum-length-excluded, edge-excluded, and accepted chord counts/means/fractions for MLI and measured non-airspace chords so filter burden is auditable.",
        },
        {
            "item": "First-class QC overlays",
            "value": "Per-image 06_qc folders include mask_overlay.png plus rejected_edge_chords_mli.png and, when measured, rejected_edge_chords_non_airspace.png to visualize segmentation and edge-filtered chords.",
        },
        {
            "item": "Area measurement method",
            "value": area_measurement_method_description(params.airspace_component_connectivity),
        },
        {
            "item": "Non-edge component area caveat",
            "value": "Use airspace_fraction for all segmented analysis-mask airspace before edge exclusion, and non_edge_airspace_fraction or non_edge_airspace_component_area_*_mask for fully internal connected-component burden. final_airspace_area_*_mask reports the area after the configured edge-handling setting; with default edge exclusion it matches the non-edge component area. Non-edge fraction and area are edge-censored, not unbiased total airspace estimates, and connected components are not individual alveoli.",
        },
        {"item": "Total chord rows in CSV exports", "value": chord_rows},
    ]
    return pd.DataFrame(notes)


def _excel_sheets(
    field_summary: pd.DataFrame,
    slide_summary: pd.DataFrame,
    all_chords: pd.DataFrame,
    log_df: pd.DataFrame,
    params,
    *,
    pixel_area_um2: float,
    calibration_is_default: bool,
    calibration_warning: str,
) -> dict[str, pd.DataFrame]:
    return {
        "Slides": _compact_export_frame(slide_summary, SLIDE_EXPORT_COLUMNS),
        "Fields": _compact_export_frame(field_summary, FIELD_EXPORT_COLUMNS),
        "Field QC": _compact_export_frame(field_summary, FIELD_QC_EXPORT_COLUMNS),
        "Run settings": _run_settings_sheet(
            params,
            pixel_area_um2=pixel_area_um2,
            calibration_is_default=calibration_is_default,
        ),
        "Run log": _compact_run_log(log_df),
        "Column dictionary": _column_dictionary(),
        "Export notes": _excel_export_notes(
            all_chords,
            params,
            pixel_area_um2=pixel_area_um2,
            calibration_warning=calibration_warning,
        ),
    }


def _audit_excel_sheets(
    field_summary: pd.DataFrame,
    slide_summary: pd.DataFrame,
    log_df: pd.DataFrame,
    parameter_record: dict[str, object],
) -> dict[str, pd.DataFrame]:
    return {
        "field_summary": field_summary,
        "slide_summary": slide_summary,
        "processing_log": log_df,
        "parameters": pd.DataFrame(
            [{"parameter": key, "value": value} for key, value in parameter_record.items()]
        ),
    }

