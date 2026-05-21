"""Command-line interface for Lingappan MLI Analyzer."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .analysis import (
    DEFAULT_CALIBRATION_SOURCE,
    DEFAULT_FIELD_EXCLUSION_CRITERIA,
    DEFAULT_FIELD_SELECTION_METHOD,
    DEFAULT_PIXEL_HEIGHT_UM,
    DEFAULT_PIXEL_WIDTH_UM,
    AnalysisParams,
    process_files,
)
from .image_validation import (
    DEFAULT_MAX_IMAGE_DIMENSION,
    DEFAULT_MAX_IMAGE_FILE_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_IMAGE_TOTAL_BYTES,
    format_bytes,
    image_batch_validation_errors,
    image_preflight_error_message,
)
from .io import discover_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lingappan-mli",
        description="Semi-automated MLI chord measurement for cropped lung histology fields.",
    )
    parser.add_argument("inputs", nargs="+", help="Image files and/or folders to process")
    parser.add_argument("--output", "-o", default="lingappan_mli_results", help="Output folder")
    parser.add_argument(
        "--pixel-width-um",
        type=float,
        default=DEFAULT_PIXEL_WIDTH_UM,
        help=f"Pixel width in micrometers (default: {DEFAULT_PIXEL_WIDTH_UM:g})",
    )
    parser.add_argument(
        "--pixel-height-um",
        type=float,
        default=DEFAULT_PIXEL_HEIGHT_UM,
        help=f"Pixel height in micrometers (default: {DEFAULT_PIXEL_HEIGHT_UM:g})",
    )
    parser.add_argument(
        "--calibration-source",
        default=DEFAULT_CALIBRATION_SOURCE,
        help=(
            "Free-text source for the pixel-size calibration, e.g. microscope metadata or scale-bar measurement. "
            "Leaving the built-in default records and logs a default-calibration warning."
        ),
    )
    parser.add_argument("--grid-strategy", choices=["count", "spacing"], default="count")
    parser.add_argument("--num-lines", type=int, default=15, help="Lines per orientation when grid-strategy=count")
    parser.add_argument("--line-spacing-um", type=float, default=35.4, help="Line spacing when grid-strategy=spacing")
    parser.add_argument(
        "--random-grid-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply seed-controlled systematic-random phase offsets to count or spacing grids "
            "(default: enabled; use --no-random-grid-offset to disable)."
        ),
    )
    parser.add_argument(
        "--grid-random-seed",
        type=int,
        default=None,
        help="Optional non-negative random-grid seed; generated and recorded when random grid phase is enabled.",
    )
    parser.add_argument("--orientation", choices=["horizontal", "vertical", "both"], default="both")
    parser.add_argument("--threshold-method", choices=["huang", "otsu"], default="huang")
    parser.add_argument("--airspace-dark", action="store_true", help="Use this if airspaces are darker than tissue")
    parser.add_argument(
        "--airspace-component-connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Connectivity for airspace connected-component area metrics (default: 8).",
    )
    parser.add_argument("--include-edge-touching", action="store_true", help="Keep chords touching image borders")
    parser.add_argument("--min-chord-um", type=float, default=0.0, help="Minimum chord length to keep")
    parser.add_argument(
        "--measure-non-airspace",
        action="store_true",
        help="Also measure non-airspace chords; this is not wall thickness.",
    )
    parser.add_argument("--slide-roi-separator", default="_", help="Separator used in filenames, e.g. slide_ROI.tif")
    parser.add_argument(
        "--field-selection-method",
        default=DEFAULT_FIELD_SELECTION_METHOD,
        help="Free-text method/protocol used upstream to select pre-cropped ROI fields; recorded in outputs.",
    )
    parser.add_argument(
        "--field-selection-notes",
        default="",
        help="Optional free-text notes about upstream field/ROI selection; recorded in outputs.",
    )
    parser.add_argument(
        "--field-exclusion-criteria",
        default=DEFAULT_FIELD_EXCLUSION_CRITERIA,
        help="Free-text criteria used upstream to exclude fields before upload; recorded in outputs.",
    )
    parser.add_argument(
        "--max-image-file-bytes",
        type=int,
        default=DEFAULT_MAX_IMAGE_FILE_BYTES,
        help=f"Per-image file-size preflight limit in bytes (default: {format_bytes(DEFAULT_MAX_IMAGE_FILE_BYTES)})",
    )
    parser.add_argument(
        "--max-image-total-bytes",
        type=int,
        default=DEFAULT_MAX_IMAGE_TOTAL_BYTES,
        help=f"Batch file-size preflight limit in bytes (default: {format_bytes(DEFAULT_MAX_IMAGE_TOTAL_BYTES)})",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=DEFAULT_MAX_IMAGE_PIXELS,
        help=f"Per-image pixel-count preflight limit (default: {DEFAULT_MAX_IMAGE_PIXELS:,})",
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=DEFAULT_MAX_IMAGE_DIMENSION,
        help=f"Maximum width or height in pixels for preflight (default: {DEFAULT_MAX_IMAGE_DIMENSION:,})",
    )
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP archive of the output folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        params = AnalysisParams(
            pixel_width_um=args.pixel_width_um,
            pixel_height_um=args.pixel_height_um,
            calibration_source=args.calibration_source,
            grid_strategy=args.grid_strategy,
            num_lines=args.num_lines,
            line_spacing_um=args.line_spacing_um,
            grid_random_offset=args.random_grid_offset,
            grid_random_seed=args.grid_random_seed,
            orientation=args.orientation,
            threshold_method=args.threshold_method,
            airspace_bright=not args.airspace_dark,
            airspace_component_connectivity=args.airspace_component_connectivity,
            exclude_edge_touching=not args.include_edge_touching,
            min_chord_um=args.min_chord_um,
            measure_non_airspace=args.measure_non_airspace,
            slide_roi_separator=args.slide_roi_separator,
            field_selection_method=args.field_selection_method,
            field_selection_notes=args.field_selection_notes,
            field_exclusion_criteria=args.field_exclusion_criteria,
        ).validated()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        images = discover_images(args.inputs)
    except OSError as exc:
        print(f"Error: could not discover input images: {exc}", file=sys.stderr)
        return 2
    if not images:
        print(
            "Error: No supported images found. Provide TIF/TIFF/PNG/JPG/BMP images or folders containing them.",
            file=sys.stderr,
        )
        return 2

    preflight_errors = image_batch_validation_errors(
        images,
        max_image_count=None,
        max_image_file_bytes=args.max_image_file_bytes,
        max_image_total_bytes=args.max_image_total_bytes,
        max_image_pixels=args.max_image_pixels,
        max_image_dimension=args.max_image_dimension,
    )
    if preflight_errors:
        print(f"Error: {image_preflight_error_message(preflight_errors)}", file=sys.stderr)
        return 2

    try:
        result = process_files(
            images,
            args.output,
            params,
            max_image_count=None,
            max_image_file_bytes=args.max_image_file_bytes,
            max_image_total_bytes=args.max_image_total_bytes,
            max_image_pixels=args.max_image_pixels,
            max_image_dimension=args.max_image_dimension,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Error: could not write analysis outputs: {exc}", file=sys.stderr)
        return 2
    field_summary = result["field_summary"]
    log_df = result["processing_log"]
    ok_count = int((log_df["status"] == "ok").sum()) if not log_df.empty else 0
    error_count = int((log_df["status"] == "error").sum()) if not log_df.empty else 0
    total_chords = int(field_summary.get("mli_chord_count", 0).sum()) if not field_summary.empty else 0
    total_non_airspace_chords = (
        int(field_summary.get("non_airspace_chord_count", 0).sum())
        if not field_summary.empty and "non_airspace_chord_count" in field_summary
        else 0
    )

    print(f"Processed fields: {ok_count}")
    print(f"Errors: {error_count}")
    print(f"MLI chords measured: {total_chords}")
    print("Slide summaries: fields are the experimental unit for field-balanced means/SEM; chord-pooled means are supplementary.")
    if params.grid_random_offset:
        print(f"Systematic-random grid phase: enabled (seed: {params.grid_random_seed})")
    else:
        print("Systematic-random grid phase: disabled")
    print(f"Calibration source: {params.calibration_source}")
    calibration_warnings: list[str] = []
    if not field_summary.empty and "calibration_warning" in field_summary:
        calibration_warnings.extend(
            str(value)
            for value in field_summary["calibration_warning"].dropna().unique()
            if str(value).strip()
        )
    if not log_df.empty and {"filename", "message"}.issubset(log_df.columns):
        calibration_warnings.extend(
            str(value)
            for value in log_df.loc[log_df["filename"] == "calibration", "message"].dropna().unique()
            if str(value).strip()
        )
    for warning in dict.fromkeys(calibration_warnings):
        print(f"WARNING: {warning}")
    if not log_df.empty and {"filename", "status", "message"}.issubset(log_df.columns):
        image_warning_rows = log_df[
            (log_df["status"] == "warning")
            & (log_df["filename"] != "calibration")
            & log_df["message"].fillna("").astype(str).str.strip().astype(bool)
        ]
        for _, row in image_warning_rows.drop_duplicates(subset=["filename", "message"]).iterrows():
            print(f"WARNING [{row['filename']}]: {row['message']}")
    print(f"Airspace component connectivity: {params.airspace_component_connectivity}-connected")
    if params.measure_non_airspace or total_non_airspace_chords:
        print(f"Non-airspace chords measured: {total_non_airspace_chords}")
        print("Non-airspace chord statistic is not septal wall thickness.")
    print(f"Results folder: {Path(args.output).resolve()}")

    if args.zip:
        try:
            archive = shutil.make_archive(str(Path(args.output).resolve()), "zip", root_dir=args.output)
        except OSError as exc:
            print(f"Error: could not create ZIP archive: {exc}", file=sys.stderr)
            return 2
        print(f"ZIP archive: {archive}")

    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
