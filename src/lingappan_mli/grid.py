"""Grid/test-line generation for MLI chord measurement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridLine:
    """A horizontal or vertical test line in image coordinates."""

    line_id: str
    orientation: str  # "Horizontal" or "Vertical"
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def position(self) -> int:
        return self.y1 if self.orientation == "Horizontal" else self.x1


@dataclass(frozen=True)
class GridLayout:
    """Generated test lines plus the phase offsets used to create them."""

    lines: list[GridLine]
    random_offset: bool
    random_seed: int | None
    horizontal_offset_px: float | None
    vertical_offset_px: float | None


def _unique_valid_positions(values: np.ndarray, upper_exclusive: int) -> list[int]:
    if upper_exclusive <= 0:
        return []
    positions = np.rint(values).astype(int)
    positions = np.clip(positions, 0, upper_exclusive - 1)
    return sorted(np.unique(positions).astype(int).tolist())


def positions_by_count(size_px: int, count: int, offset_px: float = 0.0) -> list[int]:
    """Generate evenly spaced positions, avoiding exact image borders by default.

    ``offset_px`` shifts the systematic grid phase while preserving the default
    centered positions when it is zero.
    """
    count = max(int(count), 0)
    if size_px <= 0 or count <= 0:
        return []
    values = np.linspace(0, size_px - 1, count + 2, dtype=float)[1:-1]
    if offset_px:
        values = values + float(offset_px)
    return _unique_valid_positions(values, size_px)


def positions_by_spacing(size_px: int, spacing_px: float, offset_px: float = 0.0) -> list[int]:
    """Generate centered positions for a physical line spacing.

    ``offset_px`` shifts the default centered phase (``spacing_px / 2``). A
    systematic-random phase can therefore be represented as a single shared
    offset applied to every line in the orientation.
    """
    if size_px <= 0:
        return []
    spacing_px = max(float(spacing_px), 1.0)
    values = np.arange(
        spacing_px / 2.0 + float(offset_px),
        size_px,
        spacing_px,
        dtype=float,
    )
    return _unique_valid_positions(values, size_px)


def _count_random_offset_px(size_px: int, count: int, rng: np.random.Generator) -> float:
    count = max(int(count), 0)
    if size_px <= 1 or count <= 0:
        return 0.0
    line_interval_px = float(size_px - 1) / float(count + 1)
    if line_interval_px <= 0:
        return 0.0
    return float(rng.uniform(-line_interval_px / 2.0, line_interval_px / 2.0))


def _spacing_random_offset_px(spacing_px: float, rng: np.random.Generator) -> float:
    spacing_px = max(float(spacing_px), 1.0)
    return float(rng.uniform(-spacing_px / 2.0, spacing_px / 2.0))


def _axis_random_offset_px(
    *,
    size_px: int,
    strategy: str,
    num_lines: int,
    spacing_px: float,
    rng: np.random.Generator,
) -> float:
    if strategy == "count":
        return _count_random_offset_px(size_px, num_lines, rng)
    return _spacing_random_offset_px(spacing_px, rng)


def generate_grid_layout(
    image_shape: tuple[int, int],
    *,
    orientation: str = "both",
    strategy: str = "count",
    num_lines: int = 15,
    line_spacing_um: float = 35.4,
    pixel_width_um: float = 0.57,
    pixel_height_um: float = 0.57,
    random_offset: bool = False,
    random_seed: int | None = None,
) -> GridLayout:
    """Generate horizontal and/or vertical test lines with offset metadata.

    ``strategy="count"`` creates ``num_lines`` per selected orientation.
    ``strategy="spacing"`` creates centered lines at ``line_spacing_um``.
    When ``random_offset`` is true, a seed-controlled systematic-random phase
    offset is sampled independently for each selected orientation and applied to
    the whole grid for that orientation.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    orientation_norm = orientation.lower().strip()
    strategy_norm = strategy.lower().strip()
    if orientation_norm not in {"horizontal", "vertical", "both"}:
        raise ValueError("orientation must be 'horizontal', 'vertical', or 'both'")
    if strategy_norm not in {"count", "spacing"}:
        raise ValueError("strategy must be 'count' or 'spacing'")
    if pixel_width_um <= 0 or pixel_height_um <= 0:
        raise ValueError("pixel size must be positive")

    rng = np.random.default_rng(random_seed) if random_offset else None
    lines: list[GridLine] = []
    horizontal_offset_px: float | None = None
    vertical_offset_px: float | None = None

    if orientation_norm in {"horizontal", "both"}:
        horizontal_spacing_px = line_spacing_um / pixel_height_um
        horizontal_offset_px = (
            _axis_random_offset_px(
                size_px=height,
                strategy=strategy_norm,
                num_lines=num_lines,
                spacing_px=horizontal_spacing_px,
                rng=rng,
            )
            if rng is not None
            else 0.0
        )
        if strategy_norm == "count":
            ys = positions_by_count(height, num_lines, offset_px=horizontal_offset_px)
        else:
            ys = positions_by_spacing(height, horizontal_spacing_px, offset_px=horizontal_offset_px)
        for idx, y in enumerate(ys, start=1):
            lines.append(
                GridLine(f"H{idx:03d}", "Horizontal", 0, int(y), max(width - 1, 0), int(y))
            )

    if orientation_norm in {"vertical", "both"}:
        vertical_spacing_px = line_spacing_um / pixel_width_um
        vertical_offset_px = (
            _axis_random_offset_px(
                size_px=width,
                strategy=strategy_norm,
                num_lines=num_lines,
                spacing_px=vertical_spacing_px,
                rng=rng,
            )
            if rng is not None
            else 0.0
        )
        if strategy_norm == "count":
            xs = positions_by_count(width, num_lines, offset_px=vertical_offset_px)
        else:
            xs = positions_by_spacing(width, vertical_spacing_px, offset_px=vertical_offset_px)
        for idx, x in enumerate(xs, start=1):
            lines.append(
                GridLine(f"V{idx:03d}", "Vertical", int(x), 0, int(x), max(height - 1, 0))
            )

    return GridLayout(
        lines=lines,
        random_offset=bool(random_offset),
        random_seed=random_seed,
        horizontal_offset_px=horizontal_offset_px,
        vertical_offset_px=vertical_offset_px,
    )


def generate_grid(
    image_shape: tuple[int, int],
    *,
    orientation: str = "both",
    strategy: str = "count",
    num_lines: int = 15,
    line_spacing_um: float = 35.4,
    pixel_width_um: float = 0.57,
    pixel_height_um: float = 0.57,
    random_offset: bool = False,
    random_seed: int | None = None,
) -> list[GridLine]:
    """Generate horizontal and/or vertical test lines.

    This convenience wrapper returns only the line list. Use
    :func:`generate_grid_layout` when the applied phase offsets are needed.
    """
    return generate_grid_layout(
        image_shape,
        orientation=orientation,
        strategy=strategy,
        num_lines=num_lines,
        line_spacing_um=line_spacing_um,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
        random_offset=random_offset,
        random_seed=random_seed,
    ).lines
