# Methods and implementation notes

## Intended input

The app expects pre-cropped lung fields/ROIs (`input_image_type=pre_cropped_lung_roi`). It does not accept or select fields from whole-slide images.

Each run records upstream field-selection metadata in `parameters.json`, `field_summary`, and `processing_log`:

- `field_selection_stage=upstream_pre_app` and `field_selected_by_app=False` document that ROI selection happened before analysis.
- `field_selection_method`, `field_selection_notes`, and `field_exclusion_criteria` are user-supplied run-level descriptors for how ROIs were selected and which fields were excluded before upload.
- `field_included_in_summary`, `field_exclusion_flag`, and `field_exclusion_reason` flag whether an uploaded image entered summaries; processing failures are flagged in `processing_log`.
- `edge_exclusion_applied` and `short_chord_exclusion_applied` audit measurement-level exclusion settings for edge-touching and minimum-length chords.

## Image-load audit

Image loading preserves the existing analysis behavior: the analyzed frame is converted to RGB uint8 before grayscale conversion and thresholding, and multi-page TIFF files are analyzed from frame/page 0 only.

- `image_original_format`, `image_original_mode`, and `image_original_dtype` record the source file format, Pillow mode, and NumPy dtype observed for the analyzed frame.
- `image_output_dtype` records the dtype passed downstream for thresholding (`uint8`).
- `image_frame_index` records the analyzed frame/page (`0`), and `image_page_count` records Pillow's page/frame count when available.
- `image_scaling_applied`, `image_scaling_min`, and `image_scaling_max` record whether non-uint8 numeric image data was min-max scaled to uint8 and the finite intensity range used.
- `image_embedded_resolution_x`, `image_embedded_resolution_y`, `image_embedded_resolution_unit`, `image_embedded_resolution_source`, `image_embedded_pixel_width_um`, and `image_embedded_pixel_height_um` record TIFF/JPEG/PNG DPI or TIFF resolution tags when available and convertible to physical pixel size.
- `image_load_warnings` records per-image warnings. The same warnings appear as `warning` rows in `processing_log`, including non-uint8 scaling, multi-page TIFF first-page use, and plausible embedded-resolution metadata that disagrees with the analysis calibration.

## Calibration audit

The default pixel calibration is `pixel_width_um=0.57` by `pixel_height_um=0.57`, but physical measurements should be reported only after users verify pixel size against microscope/camera metadata, a stage micrometer, or a scale bar. Each run can record a free-text `calibration_source` through `AnalysisParams`, the CLI (`--calibration-source`), or the App calibration step. Embedded image DPI/resolution metadata is audited and compared against the selected calibration when it implies a plausible microscopy pixel size, but the app does not automatically override the user-supplied calibration because DPI metadata is often generic or scanner-written.

Outputs record:

- `pixel_width_um`, `pixel_height_um`, and `pixel_area_um2`.
- `calibration_source`, `calibration_is_default`, and `calibration_warning`.
- `width_px`, `height_px`, plus `field_width_um`, `field_height_um`, and `field_area_um2` for each analyzed field.

When the built-in 0.57 × 0.57 µm/px default remains in use with the default source label, `processing_log` receives a run-level warning and the field summaries carry the same warning text. When embedded image metadata implies a different plausible pixel size, the field's `image_load_warnings` and `processing_log` receive a calibration-mismatch warning.

## Batch field-size QC

After successful field processing, the app checks whether a run contains mixed analyzed pixel dimensions or physical field dimensions that vary by more than 2% after applying calibration. The warnings appear as `batch_qc` rows in `processing_log` and in the App/CLI run status so users can confirm that all fields used comparable magnification, calibration, and crop protocols before interpreting slide summaries.

## Default MLI pipeline

1. Load the first image/page and convert to RGB uint8, preserving existing first-frame TIFF behavior.
2. Record image pixel dimensions, original image format/mode/dtype, output dtype, analyzed frame/page index, page count when available, non-uint8 scaling min/max, embedded resolution metadata when available, image-load warnings, calibration source, pixel area, and physical field dimensions.
3. Convert RGB to 8-bit grayscale.
4. Apply Huang thresholding.
5. Classify bright pixels above threshold as airspace by default.
6. Quantify total airspace area, non-edge/fully internal connected-component area, and largest-component QC metrics from filled analysis-mask pixel counts using 8-connected components by default.
7. Generate systematic test lines:
   - fixed count per orientation, or
   - fixed physical spacing in micrometers.
   Systematic-random phase offsets can shift the count or spacing grid by a seed-controlled amount; the App and CLI enable this by default, while library/core defaults remain deterministic and centered unless random offsets are requested.
8. Place one-pixel-wide horizontal and/or vertical test lines across the airspace mask.
9. Treat each uninterrupted airspace segment along a test line as one chord.
10. Exclude chords touching the image border by default.
11. Convert chord length from pixels to micrometers.
12. Summarize chord lengths by field and by inferred slide ID.
13. Report slide-level weighting explicitly: field-balanced means/SD/SEM use fields as the experimental unit, supplementary chord-pooled means weight fields by accepted chord count, and mean chords per field reports the chord-count burden behind that weighting.
14. Report raw, include-edge, minimum-length-excluded, edge-excluded, and accepted chord counts/means/fractions to quantify filter sensitivity.
15. Report MLI summaries, horizontal/vertical diagnostics, and consistency checks from total sampled airspace length and boundary-intersection counts when complete boundary pairs are present.

## Chord measurement

The app first converts the image into an analysis mask in which each pixel is classified as airspace or non-airspace. Huang thresholding is used by default, and the selected threshold method is recorded in the outputs.

The app then places one-pixel-wide horizontal and/or vertical test lines across that mask. Along each line, every uninterrupted airspace segment is one chord. The app measures each accepted chord in pixels and converts the length to micrometers using the supplied pixel calibration.

## MLI and non-airspace chord outputs

- MLI chords are contiguous **airspace** runs along test lines.
- Non-airspace chords are contiguous **non-airspace** runs along the same test lines when enabled.
- `mean_non_airspace_chord_um` and `orientation_balanced_mean_non_airspace_chord_um` report the orientation-balanced direct mean of accepted non-airspace chords; `pooled_mean_non_airspace_chord_um` reports the pooled accepted non-airspace chord mean.
- `mli_orientation_balanced_mean_um` and `non_airspace_orientation_balanced_mean_um` average the horizontal and vertical direct chord means when both orientations are used.
- `mli_mean_um` and `non_airspace_mean_um` are pooled direct chord means.
- `mli_pooled_chord_mean_um`/`non_airspace_pooled_chord_mean_um` and `mli_orientation_balanced_direct_mean_um`/`non_airspace_orientation_balanced_direct_mean_um` are clearer field-level aliases that distinguish pooled chord means from orientation-balanced direct means.
- In `slide_summary`, `field_balanced_mean_*`, `field_sd_*`, and `field_sem_*` columns use fields as the experimental unit. `chord_pooled_mean_*` columns are supplementary descriptive means that pool accepted chords across fields and therefore weight fields with more chords more heavily; `mean_*_chords_per_field` reports this weighting burden.
- `mli_direct_horizontal_vertical_mean_ratio` and `mli_direct_horizontal_vertical_mean_delta_um` report the horizontal direct mean divided by the vertical direct mean, and the horizontal direct mean minus the vertical direct mean, when both orientations have accepted chords. Matching `non_airspace_*` columns are emitted when non-airspace chords are measured.
- `mli_direct_mean_um`, `mli_direct_orientation_balanced_mean_um`, and orientation-specific `mli_direct_*_mean_um` columns make explicit which summaries are direct chord means.
- `mli_chord_count_raw`/`non_airspace_chord_count_raw` report phase runs before edge/minimum-length filtering; `mli_chord_count_include_edge`/`non_airspace_chord_count_include_edge` report runs after minimum-length filtering while retaining edge-touching chords.
- `mli_chord_count_min_length_excluded`/`non_airspace_chord_count_min_length_excluded` and `mli_chord_fraction_min_length_excluded`/`non_airspace_chord_fraction_min_length_excluded` quantify the count and fraction of raw candidates removed by the configured minimum-length filter.
- `mli_chord_count_edge_excluded`/`non_airspace_chord_count_edge_excluded` and `mli_chord_fraction_edge_excluded`/`non_airspace_chord_fraction_edge_excluded` quantify the count and fraction of include-edge candidates removed by the configured edge filter.
- `mli_chord_count`/`non_airspace_chord_count` and `mli_chord_count_accepted`/`non_airspace_chord_count_accepted` report accepted chords after final filtering.
- `mli_mean_um_raw`/`non_airspace_mean_um_raw`, `mli_mean_um_include_edge`/`non_airspace_mean_um_include_edge`, and `mli_mean_um_accepted`/`non_airspace_mean_um_accepted` are pooled raw/include-edge/accepted means. Matching `*_orientation_balanced_mean_um_raw`, `*_orientation_balanced_mean_um_include_edge`, and `*_orientation_balanced_mean_um_accepted` columns provide orientation-balanced sensitivity means.
- `mli_total_airspace_line_length_um_raw` and `mli_total_airspace_line_length_um_accepted` report summed raw/accepted airspace length sampled by the test lines.
- `mli_airspace_boundary_intersection_count_raw` and `mli_airspace_boundary_intersection_count_accepted` report in-line airspace/non-airspace boundary crossings before and after chord filters. Contacts with the image/test-line edge are not counted as observed boundaries.
- `mli_indirect_equivalent_*mean_um` values are consistency metrics: when accepted chords have complete paired in-line boundaries, they equal `2 × accepted airspace line length / accepted boundary intersections`. They are blank when that equivalence is not applicable, such as when accepted chords touch the image edge.

## Airspace area outputs

The app reports filled foreground-pixel counts converted by pixel area. Each field summary and `parameters.json` record `area_measurement_method=filled_connected_component_pixel_count`, the configured `area_measurement_connectivity` (`8_connected` by default, `4_connected` when requested), and `area_column_semantics=filled_connected_component_pixel_counts`. The analysis mask is the raw thresholded airspace mask.

- `airspace_fraction`: all segmented airspace fraction before edge exclusion over the full image.
- `non_edge_airspace_fraction`: fraction of the full image occupied by fully internal/non-edge airspace components after removing edge-touching components.
- `total_airspace_area_um2_mask`: all analysis-mask airspace pixels × pixel area.
- `final_airspace_area_um2_mask`: airspace component area after configured edge handling; with default edge exclusion this equals the non-edge component area below.
- `non_edge_airspace_component_area_um2_mask`: fully internal/non-edge filled connected-component area.
- `largest_airspace_component_area_um2` and `largest_airspace_component_fraction_of_airspace`: QC diagnostics reporting the absolute and fractional size of the dominant segmented airspace component; `largest_airspace_component_touches_edge` records whether it contacts the field boundary.

The non-edge fraction and component-area columns are censored by the exclusion of components touching the image boundary. Use the `total_airspace_area_*` and all-airspace `airspace_fraction*` columns when the question is gross segmented airspace burden. Connected components are not individual alveoli.

## Main outputs

- `lingappan_mli_results.xlsx`: formatted Excel workbook with compact, human-readable `Slides`, `Fields`, `Field QC`, `Run settings`, `Run log`, `Column dictionary`, and `Export notes` sheets.
- `audit/lingappan_mli_audit.xlsx`: complete machine-readable field, slide, and processing-log audit workbook.
- `all_chords.csv`: compact, human-readable chord table for all images.
- `audit/all_chords_audit.csv`: complete machine-readable chord-level table, including line ID, start/end coordinates, pixel length, and audit fields.
- `parameters.json`: full reproducibility record of analysis settings, including calibration source/default-warning metadata, upstream field-selection/exclusion metadata, and configured area-measurement connectivity metadata.
- `audit/lingappan_mli_audit.xlsx` (`field_summary` sheet), per-image `summary.json`, and `processing_log`: image-load audit metadata, including original mode/dtype, analyzed frame/page index and page count when available, non-uint8 scaling min/max, embedded resolution metadata, and load/resolution warnings.
- Per-image folders containing preprocessing, contour QC, MLI outputs, non-airspace chord outputs, chord CSVs under `05_data/` only, and first-class mask/rejected-edge overlays under `06_qc/`.
