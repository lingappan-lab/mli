"""Connected-component airspace area quantification utilities.

This module intentionally avoids OpenCV so the public app remains lightweight
and easy to deploy. Airspace components are connected components in the binary
mask, and component area is reported as the filled pixel-mask area rather than
geometric contour integration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Union

import numpy as np

from .visualization import mask_to_uint8

AREA_MEASUREMENT_METHOD = "filled_connected_component_pixel_count"
AREA_COLUMN_SEMANTICS = "filled_connected_component_pixel_counts"
DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY = 8


def normalize_airspace_component_connectivity(
    connectivity: int | str = DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY,
) -> int:
    """Return a supported airspace-component connectivity (4 or 8)."""
    try:
        normalized = int(connectivity)
    except (TypeError, ValueError):
        text = str(connectivity).strip().lower().replace("-", "_")
        if text.startswith("4"):
            normalized = 4
        elif text.startswith("8"):
            normalized = 8
        else:
            raise ValueError("airspace_component_connectivity must be 4 or 8") from None
    if normalized not in {4, 8}:
        raise ValueError("airspace_component_connectivity must be 4 or 8")
    return normalized


def airspace_component_connectivity_label(
    connectivity: int | str = DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY,
) -> str:
    """Return the metadata label for a supported component connectivity."""
    return f"{normalize_airspace_component_connectivity(connectivity)}_connected"


def area_measurement_method_description(connectivity: int | str = DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY) -> str:
    """Return the filled-component area-method description for this connectivity."""
    connectivity_label = airspace_component_connectivity_label(connectivity).replace("_", "-")
    return (
        f"Areas are sums of filled {connectivity_label} airspace-component foreground pixels "
        "converted by pixel area; they are not produced by geometric contour integration."
    )


AreaMetricValue = Union[float, int, str, bool, None]


@dataclass(frozen=True)
class Component:
    rows: np.ndarray
    cols: np.ndarray
    touches_edge: bool

    @property
    def area_px(self) -> int:
        return int(self.rows.size)


@dataclass(frozen=True)
class AirspaceContourAnalysis:
    """Connected-component airspace area metrics, method metadata, and QC images."""

    metrics: dict[str, AreaMetricValue]
    all_contours_rgb: np.ndarray
    final_contours_rgb: np.ndarray
    final_contours_filled_rgb: np.ndarray


def _neighbor_offsets(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (0, -1), (0, 1), (1, 0)]
    return [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]


def _connected_components(
    mask: np.ndarray,
    *,
    connectivity: int | str = DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY,
) -> list[Component]:
    """Return connected components from a boolean mask using 4- or 8-connectivity."""
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: list[Component] = []
    neighbors = _neighbor_offsets(normalize_airspace_component_connectivity(connectivity))

    starts = np.argwhere(binary & ~visited)
    for start_row, start_col in starts:
        sr = int(start_row)
        sc = int(start_col)
        if visited[sr, sc] or not binary[sr, sc]:
            continue
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        visited[sr, sc] = True
        rows: list[int] = []
        cols: list[int] = []
        touches_edge = sr == 0 or sc == 0 or sr == height - 1 or sc == width - 1

        while queue:
            row, col = queue.popleft()
            rows.append(row)
            cols.append(col)
            if row == 0 or col == 0 or row == height - 1 or col == width - 1:
                touches_edge = True
            for dr, dc in neighbors:
                nr = row + dr
                nc = col + dc
                if nr < 0 or nc < 0 or nr >= height or nc >= width:
                    continue
                if visited[nr, nc] or not binary[nr, nc]:
                    continue
                visited[nr, nc] = True
                queue.append((nr, nc))

        components.append(
            Component(
                rows=np.asarray(rows, dtype=np.int32),
                cols=np.asarray(cols, dtype=np.int32),
                touches_edge=bool(touches_edge),
            )
        )
    return components


def _base_rgb(mask: np.ndarray) -> np.ndarray:
    binary = mask_to_uint8(mask)
    return np.stack([binary, binary, binary], axis=-1)


def _component_boundary(component: Component, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column arrays for the component boundary pixels."""
    rows = component.rows
    cols = component.cols
    height, width = mask.shape
    boundary = np.zeros(rows.shape, dtype=bool)
    for idx, (row, col) in enumerate(zip(rows, cols)):
        row_i = int(row)
        col_i = int(col)
        if row_i == 0 or col_i == 0 or row_i == height - 1 or col_i == width - 1:
            boundary[idx] = True
            continue
        if (
            not mask[row_i - 1, col_i]
            or not mask[row_i + 1, col_i]
            or not mask[row_i, col_i - 1]
            or not mask[row_i, col_i + 1]
        ):
            boundary[idx] = True
    return rows[boundary], cols[boundary]


def _area_metric_entries(area_key: str, area_px: int, pixel_area_um2: float) -> dict[str, AreaMetricValue]:
    return {
        f"{area_key}_pixels_mask": int(area_px),
        f"{area_key}_um2_mask": float(area_px * pixel_area_um2),
    }


def analyze_airspace_contours(
    airspace_mask: np.ndarray,
    *,
    pixel_width_um: float,
    pixel_height_um: float,
    exclude_edge_touching: bool = True,
    connectivity: int | str = DEFAULT_AIRSPACE_COMPONENT_CONNECTIVITY,
) -> AirspaceContourAnalysis:
    """Measure connected airspace-component areas and create QC contour images.

    True pixels in ``airspace_mask`` are grouped as connected components using
    configurable 4- or 8-connectivity (default 8). Areas are filled component
    pixel counts converted to µm², not results from geometric contour integration.
    Edge-touching components are excluded from filtered/final area by default
    because their full extent is censored by the image field boundary. Components
    are segmentation objects, not individual alveoli.
    """
    normalized_connectivity = normalize_airspace_component_connectivity(connectivity)
    measurement_connectivity = airspace_component_connectivity_label(normalized_connectivity)
    mask = np.asarray(airspace_mask, dtype=bool)
    components = _connected_components(mask, connectivity=normalized_connectivity)
    non_edge_components = [c for c in components if not c.touches_edge]
    final_components = non_edge_components if exclude_edge_touching else components
    edge_components = [c for c in components if exclude_edge_touching and c.touches_edge]

    total_area_px = int(sum(c.area_px for c in components))
    final_area_px = int(sum(c.area_px for c in final_components))
    non_edge_area_px = int(sum(c.area_px for c in non_edge_components))
    largest_component = max(components, key=lambda component: component.area_px, default=None)
    largest_component_area_px = 0 if largest_component is None else int(largest_component.area_px)
    largest_component_fraction = (
        None if total_area_px <= 0 else float(largest_component_area_px / total_area_px)
    )
    largest_component_touches_edge = None if largest_component is None else bool(largest_component.touches_edge)
    pixel_area_um2 = float(pixel_width_um * pixel_height_um)
    largest_component_area_um2 = float(largest_component_area_px * pixel_area_um2)
    metrics: dict[str, AreaMetricValue] = {
        "airspace_component_connectivity": int(normalized_connectivity),
        "area_measurement_method": AREA_MEASUREMENT_METHOD,
        "area_measurement_connectivity": measurement_connectivity,
        "area_column_semantics": AREA_COLUMN_SEMANTICS,
        "airspace_component_count_all": int(len(components)),
        "airspace_component_count_final": int(len(final_components)),
        "airspace_component_count_edge_excluded": int(len(edge_components)),
        "largest_airspace_component_area_pixels": int(largest_component_area_px),
        "largest_airspace_component_area_um2": largest_component_area_um2,
        "largest_airspace_component_fraction_of_airspace": largest_component_fraction,
        "largest_airspace_component_touches_edge": largest_component_touches_edge,
        **_area_metric_entries("total_airspace_area", total_area_px, pixel_area_um2),
        **_area_metric_entries("final_airspace_area", final_area_px, pixel_area_um2),
        **_area_metric_entries("non_edge_airspace_component_area", non_edge_area_px, pixel_area_um2),
    }

    all_img = _base_rgb(mask)
    final_img = _base_rgb(mask)
    filled_img = _base_rgb(mask)

    for component in components:
        rr, cc = _component_boundary(component, mask)
        all_img[rr, cc] = [255, 0, 0]
    for component in final_components:
        rr, cc = _component_boundary(component, mask)
        final_img[rr, cc] = [255, 0, 0]
        filled_img[component.rows, component.cols] = [255, 0, 0]
    for component in edge_components:
        rr, cc = _component_boundary(component, mask)
        final_img[rr, cc] = [0, 80, 255]

    return AirspaceContourAnalysis(
        metrics=metrics,
        all_contours_rgb=all_img.astype(np.uint8, copy=False),
        final_contours_rgb=final_img.astype(np.uint8, copy=False),
        final_contours_filled_rgb=filled_img.astype(np.uint8, copy=False),
    )
