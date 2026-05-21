"""Chord measurement along systematic test lines."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grid import GridLine


@dataclass(frozen=True)
class PhaseRun:
    orientation: str
    orientation_key: str
    line_id: str
    line_position_px: int
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    length_px: int
    length_um: float
    edge_touching: bool
    boundary_count: int


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int, bool]]:
    """Return inclusive start/end indices and whether a run touches a line edge."""
    values = np.asarray(mask_1d, dtype=bool).astype(np.int16, copy=False)
    if values.size == 0:
        return []
    padded = np.concatenate((np.array([0], dtype=np.int16), values, np.array([0], dtype=np.int16)))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends_exclusive = np.flatnonzero(changes == -1)
    out: list[tuple[int, int, bool]] = []
    last_index = values.size - 1
    for start, end_excl in zip(starts, ends_exclusive):
        end = int(end_excl - 1)
        start = int(start)
        touches_edge = start == 0 or end == last_index
        if end >= start:
            out.append((start, end, touches_edge))
    return out


def _run_boundary_intersection_count(start: int, end: int, last_index: int) -> int:
    """Count in-line phase-boundary intersections for one run.

    Interior runs have two measured phase-boundary crossings. Runs touching one
    image/test-line edge have one in-line crossing, and runs spanning the full
    line have zero. Edge contacts are not counted as tissue/airspace boundary
    intersections because no boundary is observed inside the sampled line.
    """
    if end < start or last_index < 0:
        return 0
    return int(start > 0) + int(end < last_index)


def _target_profile_for_line(
    mask: np.ndarray,
    line: GridLine,
    *,
    target_airspace: bool,
    pixel_width_um: float,
    pixel_height_um: float,
) -> tuple[np.ndarray, float] | None:
    """Return the selected phase profile and micrometer scale for a test line."""
    if line.orientation == "Horizontal":
        y = int(line.y1)
        x1 = int(min(line.x1, line.x2))
        x2 = int(max(line.x1, line.x2))
        if y < 0 or y >= mask.shape[0]:
            return None
        values = mask[y, x1 : x2 + 1]
        target = values if target_airspace else ~values
        return np.asarray(target, dtype=bool), float(pixel_width_um)

    x = int(line.x1)
    y1 = int(min(line.y1, line.y2))
    y2 = int(max(line.y1, line.y2))
    if x < 0 or x >= mask.shape[1]:
        return None
    values = mask[y1 : y2 + 1, x]
    target = values if target_airspace else ~values
    return np.asarray(target, dtype=bool), float(pixel_height_um)


def _chord_coordinates_for_line(line: GridLine, start: int, end: int) -> tuple[int, int, int, int, int]:
    """Return line position plus start/end coordinates for a run on a test line."""
    if line.orientation == "Horizontal":
        y = int(line.y1)
        x1 = int(min(line.x1, line.x2))
        return y, int(x1 + start), y, int(x1 + end), y

    x = int(line.x1)
    y1 = int(min(line.y1, line.y2))
    return x, x, int(y1 + start), x, int(y1 + end)


def _iter_phase_runs(
    mask: np.ndarray,
    lines: Sequence[GridLine],
    *,
    target_airspace: bool,
    pixel_width_um: float,
    pixel_height_um: float,
) -> Iterator[PhaseRun]:
    """Yield measured runs from the phase profile for each valid test line."""
    for line in lines:
        profile_and_scale = _target_profile_for_line(
            mask,
            line,
            target_airspace=target_airspace,
            pixel_width_um=pixel_width_um,
            pixel_height_um=pixel_height_um,
        )
        if profile_and_scale is None:
            continue
        target, scale_um = profile_and_scale
        orientation_key = "horizontal" if line.orientation == "Horizontal" else "vertical"
        last_index = int(target.size - 1)
        for start, end, touches in _runs(target):
            length_px = int(end - start + 1)
            length_um = float(length_px * scale_um)
            line_position_px, start_x, start_y, end_x, end_y = _chord_coordinates_for_line(line, start, end)
            yield PhaseRun(
                orientation=line.orientation,
                orientation_key=orientation_key,
                line_id=line.line_id,
                line_position_px=line_position_px,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                length_px=length_px,
                length_um=length_um,
                edge_touching=bool(touches),
                boundary_count=_run_boundary_intersection_count(start, end, last_index),
            )


def measure_phase_chords(
    airspace_mask: np.ndarray,
    lines: Sequence[GridLine],
    *,
    phase: str = "airspace",
    pixel_width_um: float = 0.57,
    pixel_height_um: float = 0.57,
    exclude_edge_touching: bool = True,
    min_chord_um: float = 0.0,
) -> pd.DataFrame:
    """Measure contiguous phase chords along horizontal/vertical test lines.

    ``phase="airspace"`` is the MLI measurement. ``phase="non_airspace"``
    measures non-airspace line-profile runs; it is not a septal wall-thickness
    measurement.
    """
    phase_norm = phase.lower().strip().replace("-", "_").replace(" ", "_")
    if phase_norm == "tissue":
        phase_norm = "non_airspace"
    if phase_norm not in {"airspace", "non_airspace"}:
        raise ValueError("phase must be 'airspace' or 'non_airspace'")
    if pixel_width_um <= 0 or pixel_height_um <= 0:
        raise ValueError("pixel size must be positive")

    mask = np.asarray(airspace_mask, dtype=bool)
    target_airspace = phase_norm == "airspace"
    records: list[dict[str, object]] = []

    measure_type = "MLI" if target_airspace else "Non-airspace"
    for run in _iter_phase_runs(
        mask,
        lines,
        target_airspace=target_airspace,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
    ):
        if exclude_edge_touching and run.edge_touching:
            continue
        if run.length_um < min_chord_um:
            continue
        records.append(
            {
                "measure_type": measure_type,
                "phase": phase_norm,
                "orientation": run.orientation,
                "line_id": run.line_id,
                "line_position_px": run.line_position_px,
                "start_x": run.start_x,
                "start_y": run.start_y,
                "end_x": run.end_x,
                "end_y": run.end_y,
                "length_px": run.length_px,
                "length_um": run.length_um,
                "edge_touching": run.edge_touching,
            }
        )

    return pd.DataFrame.from_records(records)


def _mean_from_totals(total_length_um: float, count: int) -> float | None:
    """Return a mean length from aggregate totals, blank when no chords exist."""
    return None if count <= 0 else float(total_length_um / count)


def _orientation_balanced_mean_from_totals(
    lengths_by_orientation: dict[str, float],
    counts_by_orientation: dict[str, int],
) -> float | None:
    """Return the mean of orientation-specific means for non-empty orientations."""
    orientation_means = [
        _mean_from_totals(lengths_by_orientation[orientation], counts_by_orientation[orientation])
        for orientation in ("horizontal", "vertical")
        if counts_by_orientation[orientation] > 0
    ]
    finite_means = [value for value in orientation_means if value is not None and np.isfinite(value)]
    return float(np.mean(finite_means)) if finite_means else None


FILTER_STAGES = ("raw", "include_edge", "edge_excluded", "accepted")
COUNT_STAGES = ("raw", "include_edge", "min_length_excluded", "edge_excluded", "accepted")
ORIENTATIONS = ("horizontal", "vertical")


def _empty_chord_filter_summary(prefix: str) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {
        f"{prefix}_chord_count_{stage}": 0 for stage in COUNT_STAGES
    }
    summary.update(
        {
            f"{prefix}_chord_fraction_min_length_excluded": None,
            f"{prefix}_chord_fraction_edge_excluded": None,
        }
    )
    summary.update({f"{prefix}_mean_um_{stage}": None for stage in FILTER_STAGES})
    summary.update({f"{prefix}_orientation_balanced_mean_um_{stage}": None for stage in FILTER_STAGES})
    for orientation in ORIENTATIONS:
        summary.update({f"{prefix}_{orientation}_count_{stage}": 0 for stage in COUNT_STAGES})
    for orientation in ORIENTATIONS:
        summary.update({f"{prefix}_{orientation}_mean_um_{stage}": None for stage in FILTER_STAGES})
    return summary


def _orientation_total(values: dict[str, float] | dict[str, int]) -> float:
    return float(values["horizontal"] + values["vertical"])


def _fraction_or_none(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _chord_filter_summary(
    prefix: str,
    *,
    counts_raw: dict[str, int],
    lengths_raw: dict[str, float],
    counts_include_edge: dict[str, int],
    lengths_include_edge: dict[str, float],
    counts_edge_excluded: dict[str, int],
    lengths_edge_excluded: dict[str, float],
    counts_accepted: dict[str, int],
    lengths_accepted: dict[str, float],
) -> dict[str, float | int | None]:
    counts_by_stage = {
        "raw": counts_raw,
        "include_edge": counts_include_edge,
        "min_length_excluded": {
            orientation: int(counts_raw[orientation] - counts_include_edge[orientation])
            for orientation in ORIENTATIONS
        },
        "edge_excluded": counts_edge_excluded,
        "accepted": counts_accepted,
    }
    lengths_by_stage = {
        "raw": lengths_raw,
        "include_edge": lengths_include_edge,
        "edge_excluded": lengths_edge_excluded,
        "accepted": lengths_accepted,
    }
    total_counts = {stage: int(_orientation_total(counts)) for stage, counts in counts_by_stage.items()}
    total_lengths = {
        stage: float(_orientation_total(lengths)) for stage, lengths in lengths_by_stage.items()
    }

    summary: dict[str, float | int | None] = {
        f"{prefix}_chord_count_{stage}": total_counts[stage] for stage in COUNT_STAGES
    }
    summary.update(
        {
            f"{prefix}_chord_fraction_min_length_excluded": _fraction_or_none(
                total_counts["min_length_excluded"],
                total_counts["raw"],
            ),
            f"{prefix}_chord_fraction_edge_excluded": _fraction_or_none(
                total_counts["edge_excluded"],
                total_counts["include_edge"],
            ),
        }
    )
    summary.update(
        {
            f"{prefix}_mean_um_{stage}": _mean_from_totals(total_lengths[stage], total_counts[stage])
            for stage in FILTER_STAGES
        }
    )
    summary.update(
        {
            f"{prefix}_orientation_balanced_mean_um_{stage}": _orientation_balanced_mean_from_totals(
                lengths_by_stage[stage],
                counts_by_stage[stage],
            )
            for stage in FILTER_STAGES
        }
    )
    for orientation in ORIENTATIONS:
        summary.update(
            {
                f"{prefix}_{orientation}_count_{stage}": int(counts_by_stage[stage][orientation])
                for stage in COUNT_STAGES
            }
        )
    for orientation in ORIENTATIONS:
        summary.update(
            {
                f"{prefix}_{orientation}_mean_um_{stage}": _mean_from_totals(
                    lengths_by_stage[stage][orientation],
                    counts_by_stage[stage][orientation],
                )
                for stage in FILTER_STAGES
            }
        )
    return summary


def _empty_line_intercept_summary(prefix: str, phase_label: str) -> dict[str, float | int | None]:
    summary = _empty_chord_filter_summary(prefix)
    summary.update(
        {
            f"{prefix}_total_{phase_label}_line_length_um_raw": 0.0,
            f"{prefix}_total_{phase_label}_line_length_um_accepted": 0.0,
            f"{prefix}_horizontal_{phase_label}_line_length_um_raw": 0.0,
            f"{prefix}_horizontal_{phase_label}_line_length_um_accepted": 0.0,
            f"{prefix}_vertical_{phase_label}_line_length_um_raw": 0.0,
            f"{prefix}_vertical_{phase_label}_line_length_um_accepted": 0.0,
            f"{prefix}_{phase_label}_boundary_intersection_count_raw": 0,
            f"{prefix}_{phase_label}_boundary_intersection_count_accepted": 0,
            f"{prefix}_horizontal_{phase_label}_boundary_intersection_count_raw": 0,
            f"{prefix}_horizontal_{phase_label}_boundary_intersection_count_accepted": 0,
            f"{prefix}_vertical_{phase_label}_boundary_intersection_count_raw": 0,
            f"{prefix}_vertical_{phase_label}_boundary_intersection_count_accepted": 0,
            f"{prefix}_indirect_equivalent_mean_um": None,
            f"{prefix}_indirect_equivalent_orientation_balanced_mean_um": None,
            f"{prefix}_indirect_equivalent_horizontal_mean_um": None,
            f"{prefix}_indirect_equivalent_vertical_mean_um": None,
        }
    )
    return summary


def _indirect_equivalent_mean(
    total_phase_line_length_um: float,
    boundary_intersection_count: int,
    chord_count: int,
) -> float | None:
    """Return the boundary-count equivalent of the direct chord mean if complete.

    The equivalence is defined only when each accepted chord contributes two
    observed phase-boundary intersections. Edge-touching chords therefore make
    the boundary-count equivalent inapplicable rather than silently changing the
    denominator.
    """
    if chord_count <= 0 or boundary_intersection_count <= 0:
        return None
    if boundary_intersection_count != 2 * chord_count:
        return None
    return float(2.0 * total_phase_line_length_um / boundary_intersection_count)


def _horizontal_vertical_ratio_delta(
    horizontal_mean: float | None,
    vertical_mean: float | None,
) -> tuple[float | None, float | None]:
    """Return H/V mean ratio and signed H-minus-V delta when both exist."""
    if horizontal_mean is None or vertical_mean is None:
        return None, None
    if not (np.isfinite(horizontal_mean) and np.isfinite(vertical_mean)):
        return None, None
    if vertical_mean == 0:
        return None, None
    return float(horizontal_mean / vertical_mean), float(horizontal_mean - vertical_mean)


MEASUREMENT_SUMMARY_SOURCES = [
    ("chord_count", "chord_count"),
    ("chord_count_accepted", "chord_count"),
    ("horizontal_count", "horizontal_count"),
    ("horizontal_count_accepted", "horizontal_count"),
    ("vertical_count", "vertical_count"),
    ("vertical_count_accepted", "vertical_count"),
    ("mean_um", "pooled_mean_um"),
    ("mean_um_accepted", "pooled_mean_um"),
    ("direct_mean_um", "pooled_mean_um"),
    ("orientation_balanced_mean_um", "orientation_balanced_mean_um"),
    ("orientation_balanced_mean_um_accepted", "orientation_balanced_mean_um"),
    ("direct_orientation_balanced_mean_um", "orientation_balanced_mean_um"),
    ("horizontal_mean_um", "horizontal_mean_um"),
    ("horizontal_mean_um_accepted", "horizontal_mean_um"),
    ("direct_horizontal_mean_um", "horizontal_mean_um"),
    ("vertical_mean_um", "vertical_mean_um"),
    ("vertical_mean_um_accepted", "vertical_mean_um"),
    ("direct_vertical_mean_um", "vertical_mean_um"),
    ("median_um", "median_um"),
    ("sd_um", "sd_um"),
    ("pooled_chord_mean_um", "pooled_mean_um"),
    ("orientation_balanced_direct_mean_um", "orientation_balanced_mean_um"),
    ("direct_horizontal_vertical_mean_ratio", "horizontal_vertical_mean_ratio"),
    ("direct_horizontal_vertical_mean_delta_um", "horizontal_vertical_mean_delta_um"),
]


def _measurement_summary_from_values(
    prefix: str,
    values: dict[str, float | int | None],
) -> dict[str, float | int | None]:
    """Expand canonical measurement values to explicit derived columns."""
    return {f"{prefix}_{suffix}": values.get(source) for suffix, source in MEASUREMENT_SUMMARY_SOURCES}


def _measurement_prefix(measure_type: str | None, *, target_airspace: bool | None = None) -> str:
    if measure_type is None:
        return "mli" if target_airspace else "non_airspace"
    normalized = measure_type.lower().strip().replace("-", "_").replace(" ", "_")
    if normalized in {"nonairspace", "non_airspace", "tissue"}:
        return "non_airspace"
    return normalized


def _measurement_label(prefix: str) -> str:
    return "MLI" if prefix == "mli" else "Non-airspace"


def _with_non_airspace_chord_aliases(
    summary: dict[str, float | int | None],
    prefix: str,
) -> dict[str, float | int | None]:
    """Add preferred non-airspace chord aliases for non-airspace summaries."""
    if prefix != "non_airspace":
        return summary
    out = dict(summary)
    out.update(
        {
            "mean_non_airspace_chord_um": out.get("non_airspace_orientation_balanced_mean_um"),
            "orientation_balanced_mean_non_airspace_chord_um": out.get(
                "non_airspace_orientation_balanced_mean_um"
            ),
            "pooled_mean_non_airspace_chord_um": out.get("non_airspace_pooled_chord_mean_um"),
        }
    )
    return out


def summarize_phase_line_intercepts(
    airspace_mask: np.ndarray,
    lines: Sequence[GridLine],
    *,
    phase: str = "airspace",
    pixel_width_um: float = 0.57,
    pixel_height_um: float = 0.57,
    exclude_edge_touching: bool = True,
    min_chord_um: float = 0.0,
    measure_type: str | None = None,
) -> dict[str, float | int | None]:
    """Summarize line-profile totals for direct/indirect consistency checks.

    Metrics are computed from the same one-pixel phase profiles used for direct
    chord measurement. ``*_raw`` values include all phase runs before edge/minimum-length filters.
    ``*_include_edge`` values apply the minimum-length filter while retaining
    edge-touching runs. ``*_min_length_excluded`` values quantify raw candidates
    removed by the minimum-length filter. ``*_edge_excluded`` values quantify the
    include-edge candidates removed by the configured edge filter, and
    ``*_accepted`` values use the same filters as the chord table. The
    ``*_indirect_equivalent_*``
    means are calculated as
    ``2 * total accepted phase-line length / accepted boundary intersections``
    only when every accepted chord has two observed in-line phase boundaries.
    """
    phase_norm = phase.lower().strip().replace("-", "_").replace(" ", "_")
    if phase_norm == "tissue":
        phase_norm = "non_airspace"
    if phase_norm not in {"airspace", "non_airspace"}:
        raise ValueError("phase must be 'airspace' or 'non_airspace'")
    if pixel_width_um <= 0 or pixel_height_um <= 0:
        raise ValueError("pixel size must be positive")
    if min_chord_um < 0:
        raise ValueError("min_chord_um cannot be negative")

    mask = np.asarray(airspace_mask, dtype=bool)

    target_airspace = phase_norm == "airspace"
    prefix = _measurement_prefix(measure_type, target_airspace=target_airspace)
    phase_label = "airspace" if target_airspace else "non_airspace"
    summary = _empty_line_intercept_summary(prefix, phase_label)

    counts_raw = {"horizontal": 0, "vertical": 0}
    counts_include_edge = {"horizontal": 0, "vertical": 0}
    counts_edge_excluded = {"horizontal": 0, "vertical": 0}
    counts_accepted = {"horizontal": 0, "vertical": 0}
    lengths_raw = {"horizontal": 0.0, "vertical": 0.0}
    lengths_include_edge = {"horizontal": 0.0, "vertical": 0.0}
    lengths_edge_excluded = {"horizontal": 0.0, "vertical": 0.0}
    lengths_accepted = {"horizontal": 0.0, "vertical": 0.0}
    boundaries_raw = {"horizontal": 0, "vertical": 0}
    boundaries_accepted = {"horizontal": 0, "vertical": 0}

    for run in _iter_phase_runs(
        mask,
        lines,
        target_airspace=target_airspace,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
    ):
        orientation_key = run.orientation_key
        counts_raw[orientation_key] += 1
        lengths_raw[orientation_key] += run.length_um
        boundaries_raw[orientation_key] += run.boundary_count

        if run.length_um < min_chord_um:
            continue
        counts_include_edge[orientation_key] += 1
        lengths_include_edge[orientation_key] += run.length_um

        if exclude_edge_touching and run.edge_touching:
            counts_edge_excluded[orientation_key] += 1
            lengths_edge_excluded[orientation_key] += run.length_um
            continue
        counts_accepted[orientation_key] += 1
        lengths_accepted[orientation_key] += run.length_um
        boundaries_accepted[orientation_key] += run.boundary_count

    total_accepted_count = int(counts_accepted["horizontal"] + counts_accepted["vertical"])
    total_raw_length = float(lengths_raw["horizontal"] + lengths_raw["vertical"])
    total_accepted_length = float(lengths_accepted["horizontal"] + lengths_accepted["vertical"])
    total_raw_boundaries = int(boundaries_raw["horizontal"] + boundaries_raw["vertical"])
    total_accepted_boundaries = int(boundaries_accepted["horizontal"] + boundaries_accepted["vertical"])

    h_indirect = _indirect_equivalent_mean(
        lengths_accepted["horizontal"],
        boundaries_accepted["horizontal"],
        counts_accepted["horizontal"],
    )
    v_indirect = _indirect_equivalent_mean(
        lengths_accepted["vertical"],
        boundaries_accepted["vertical"],
        counts_accepted["vertical"],
    )
    orientation_indirect_inputs = [
        (counts_accepted["horizontal"], h_indirect),
        (counts_accepted["vertical"], v_indirect),
    ]
    if any(count > 0 and value is None for count, value in orientation_indirect_inputs):
        balanced_indirect = None
    else:
        orientation_indirect_values = [
            value for count, value in orientation_indirect_inputs if count > 0 and value is not None and np.isfinite(value)
        ]
        balanced_indirect = float(np.mean(orientation_indirect_values)) if orientation_indirect_values else None

    summary.update(
        {
            **_chord_filter_summary(
                prefix,
                counts_raw=counts_raw,
                lengths_raw=lengths_raw,
                counts_include_edge=counts_include_edge,
                lengths_include_edge=lengths_include_edge,
                counts_edge_excluded=counts_edge_excluded,
                lengths_edge_excluded=lengths_edge_excluded,
                counts_accepted=counts_accepted,
                lengths_accepted=lengths_accepted,
            ),
            f"{prefix}_total_{phase_label}_line_length_um_raw": total_raw_length,
            f"{prefix}_total_{phase_label}_line_length_um_accepted": total_accepted_length,
            f"{prefix}_horizontal_{phase_label}_line_length_um_raw": float(lengths_raw["horizontal"]),
            f"{prefix}_horizontal_{phase_label}_line_length_um_accepted": float(
                lengths_accepted["horizontal"]
            ),
            f"{prefix}_vertical_{phase_label}_line_length_um_raw": float(lengths_raw["vertical"]),
            f"{prefix}_vertical_{phase_label}_line_length_um_accepted": float(
                lengths_accepted["vertical"]
            ),
            f"{prefix}_{phase_label}_boundary_intersection_count_raw": total_raw_boundaries,
            f"{prefix}_{phase_label}_boundary_intersection_count_accepted": total_accepted_boundaries,
            f"{prefix}_horizontal_{phase_label}_boundary_intersection_count_raw": int(
                boundaries_raw["horizontal"]
            ),
            f"{prefix}_horizontal_{phase_label}_boundary_intersection_count_accepted": int(
                boundaries_accepted["horizontal"]
            ),
            f"{prefix}_vertical_{phase_label}_boundary_intersection_count_raw": int(
                boundaries_raw["vertical"]
            ),
            f"{prefix}_vertical_{phase_label}_boundary_intersection_count_accepted": int(
                boundaries_accepted["vertical"]
            ),
            f"{prefix}_indirect_equivalent_mean_um": _indirect_equivalent_mean(
                total_accepted_length,
                total_accepted_boundaries,
                total_accepted_count,
            ),
            f"{prefix}_indirect_equivalent_orientation_balanced_mean_um": balanced_indirect,
            f"{prefix}_indirect_equivalent_horizontal_mean_um": h_indirect,
            f"{prefix}_indirect_equivalent_vertical_mean_um": v_indirect,
        }
    )
    return summary


def summarize_phase_chord_filter_sensitivity(
    airspace_mask: np.ndarray,
    lines: Sequence[GridLine],
    *,
    phase: str = "airspace",
    pixel_width_um: float = 0.57,
    pixel_height_um: float = 0.57,
    exclude_edge_touching: bool = True,
    min_chord_um: float = 0.0,
    measure_type: str | None = None,
) -> dict[str, float | int | None]:
    """Summarize raw/include-edge/min-length/edge-excluded/accepted chord counts and means."""
    phase_norm = phase.lower().strip().replace("-", "_").replace(" ", "_")
    if phase_norm == "tissue":
        phase_norm = "non_airspace"
    target_airspace = phase_norm == "airspace"
    prefix = _measurement_prefix(measure_type, target_airspace=target_airspace)
    sensitivity_keys = _empty_chord_filter_summary(prefix).keys()
    full_summary = summarize_phase_line_intercepts(
        airspace_mask,
        lines,
        phase=phase,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
        exclude_edge_touching=exclude_edge_touching,
        min_chord_um=min_chord_um,
        measure_type=measure_type,
    )
    return {key: full_summary[key] for key in sensitivity_keys}


def summarize_measurements(chords: pd.DataFrame, measure_type: str) -> dict[str, float | int | None]:
    """Summarize accepted direct chord measurements for one measure type.

    The ``*_direct_*`` columns make explicit that these values are direct means
    of accepted chord lengths, not independent total-test-line intercept estimates.
    Additional explicit aliases distinguish the pooled chord mean from the
    orientation-balanced direct mean, and H/V ratio/delta columns diagnose
    orientation anisotropy when both orientations have accepted chords. For
    non-airspace measurements, preferred aliases clarify that these are
    non-airspace chord statistics, not wall-thickness measurements.
    """
    prefix = _measurement_prefix(measure_type)
    empty_summary = _measurement_summary_from_values(
        prefix,
        {
            "chord_count": 0,
            "horizontal_count": 0,
            "vertical_count": 0,
            "pooled_mean_um": None,
            "orientation_balanced_mean_um": None,
            "horizontal_mean_um": None,
            "vertical_mean_um": None,
            "median_um": None,
            "sd_um": None,
            "horizontal_vertical_mean_ratio": None,
            "horizontal_vertical_mean_delta_um": None,
        },
    )
    if chords.empty:
        return _with_non_airspace_chord_aliases(empty_summary, prefix)

    measure_label = _measurement_label(prefix)
    subset = chords[chords["measure_type"] == measure_label].copy()
    if subset.empty:
        return _with_non_airspace_chord_aliases(empty_summary, prefix)

    horiz = subset[subset["orientation"] == "Horizontal"]
    vert = subset[subset["orientation"] == "Vertical"]
    h_mean = float(horiz["length_um"].mean()) if not horiz.empty else None
    v_mean = float(vert["length_um"].mean()) if not vert.empty else None
    orientation_means = [value for value in (h_mean, v_mean) if value is not None and np.isfinite(value)]
    balanced = float(np.mean(orientation_means)) if orientation_means else None
    pooled_mean = float(subset["length_um"].mean())
    hv_ratio, hv_delta = _horizontal_vertical_ratio_delta(h_mean, v_mean)

    summary = _measurement_summary_from_values(
        prefix,
        {
            "chord_count": int(len(subset)),
            "horizontal_count": int(len(horiz)),
            "vertical_count": int(len(vert)),
            "pooled_mean_um": pooled_mean,
            "orientation_balanced_mean_um": balanced,
            "horizontal_mean_um": h_mean,
            "vertical_mean_um": v_mean,
            "median_um": float(subset["length_um"].median()),
            "sd_um": float(subset["length_um"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "horizontal_vertical_mean_ratio": hv_ratio,
            "horizontal_vertical_mean_delta_um": hv_delta,
        },
    )
    return _with_non_airspace_chord_aliases(summary, prefix)
