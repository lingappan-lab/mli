"""Visualization helpers for QC overlays."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .grid import GridLine


CLINICAL_BLUE = (42, 111, 219, 180)
MLI_AMBER = (245, 184, 0, 235)
NON_AIRSPACE_MAGENTA = (190, 86, 255, 220)
MASK_CYAN = (0, 180, 255, 105)
REJECTED_EDGE_RED = (235, 45, 45, 245)
TEXT_DARK = (16, 24, 40, 255)
WHITE = (255, 255, 255, 255)


def mask_to_uint8(airspace_mask: np.ndarray) -> np.ndarray:
    """Convert a boolean airspace mask to displayable uint8 image."""
    return (np.asarray(airspace_mask, dtype=bool) * 255).astype(np.uint8)


def _with_title(image: Image.Image, title: str | None) -> Image.Image:
    if not title:
        return image.convert("RGB")
    composed = image.convert("RGBA")
    banner_height = 34
    banner = Image.new("RGBA", (composed.width, banner_height), WHITE)
    banner_draw = ImageDraw.Draw(banner)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    banner_draw.text((12, 9), title, fill=TEXT_DARK, font=font)
    out = Image.new("RGBA", (composed.width, composed.height + banner_height), WHITE)
    out.paste(banner, (0, 0))
    out.paste(composed, (0, banner_height))
    return out.convert("RGB")


def _measure_type_key(value: object) -> str:
    return str(value or "MLI").strip().lower().replace("-", "_").replace(" ", "_")


def _iter_overlay_chords(
    chords: pd.DataFrame,
    *,
    include_non_airspace: bool,
    measure_types: set[str] | None,
):
    allowed = {_measure_type_key(measure_type) for measure_type in measure_types} if measure_types else None
    for _, row in chords.iterrows():
        measure_type_key = _measure_type_key(row.get("measure_type", "MLI"))
        if allowed is not None and measure_type_key not in allowed:
            continue
        if allowed is None and measure_type_key == "non_airspace" and not include_non_airspace:
            continue
        yield row, measure_type_key


def _chords_for_measure_orientation(chords: pd.DataFrame, measure_type: str, orientation: str) -> pd.DataFrame:
    if chords.empty:
        return chords
    measure_type_key = _measure_type_key(measure_type)
    return chords[
        (chords["measure_type"].map(_measure_type_key) == measure_type_key)
        & (chords["orientation"].str.lower() == orientation.lower())
    ]


def make_overlay(
    rgb: np.ndarray,
    lines: Sequence[GridLine],
    chords: pd.DataFrame,
    *,
    title: str | None = None,
    include_non_airspace: bool = False,
    measure_types: set[str] | None = None,
) -> Image.Image:
    """Create a clinical-style overlay with test grid and measured chords."""
    base = Image.fromarray(rgb.astype(np.uint8, copy=False)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw test grid first.
    for line in lines:
        draw.line((line.x1, line.y1, line.x2, line.y2), fill=CLINICAL_BLUE, width=1)

    # Draw measured MLI/non-airspace chords as thicker colored segments.
    if not chords.empty:
        for row, measure_type in _iter_overlay_chords(
            chords,
            include_non_airspace=include_non_airspace,
            measure_types=measure_types,
        ):
            color = NON_AIRSPACE_MAGENTA if measure_type == "non_airspace" else MLI_AMBER
            width = 3 if measure_type == "mli" else 2
            draw.line(
                (
                    int(row["start_x"]),
                    int(row["start_y"]),
                    int(row["end_x"]),
                    int(row["end_y"]),
                ),
                fill=color,
                width=width,
            )

    composed = Image.alpha_composite(base, overlay)
    return _with_title(composed, title)


def make_mask_overlay(
    rgb: np.ndarray,
    airspace_mask: np.ndarray,
    *,
    title: str | None = None,
) -> Image.Image:
    """Overlay the analysis airspace mask on the original field for QC."""
    rgb_array = rgb.astype(np.uint8, copy=False)
    mask = np.asarray(airspace_mask, dtype=bool)
    if tuple(mask.shape) != tuple(rgb_array.shape[:2]):
        raise ValueError("airspace_mask shape must match rgb image shape")

    base = Image.fromarray(rgb_array).convert("RGBA")
    overlay_array = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay_array[mask] = MASK_CYAN

    composed = Image.alpha_composite(base, Image.fromarray(overlay_array))
    return _with_title(composed, title)


def make_rejected_edge_chord_overlay(
    rgb: np.ndarray,
    lines: Sequence[GridLine],
    rejected_chords: pd.DataFrame,
    *,
    title: str | None = None,
) -> Image.Image:
    """Create an overlay that highlights edge-touching chords rejected by filtering."""
    base = Image.fromarray(rgb.astype(np.uint8, copy=False)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for line in lines:
        draw.line((line.x1, line.y1, line.x2, line.y2), fill=CLINICAL_BLUE, width=1)

    if not rejected_chords.empty:
        for _, row in rejected_chords.iterrows():
            draw.line(
                (
                    int(row["start_x"]),
                    int(row["start_y"]),
                    int(row["end_x"]),
                    int(row["end_y"]),
                ),
                fill=REJECTED_EDGE_RED,
                width=4,
            )

    composed = Image.alpha_composite(base, overlay)
    return _with_title(composed, title)


def make_test_line_image(shape: tuple[int, int], lines: Sequence[GridLine], orientation: str) -> Image.Image:
    """Create a white image with red one-pixel test lines, mirroring Fiji inputs."""
    height, width = int(shape[0]), int(shape[1])
    orientation_norm = orientation.lower()
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for line in lines:
        if orientation_norm != "both" and line.orientation.lower() != orientation_norm:
            continue
        draw.line((line.x1, line.y1, line.x2, line.y2), fill=(255, 0, 0), width=1)
    return image


def make_particle_image(
    shape: tuple[int, int],
    chords: pd.DataFrame,
    *,
    measure_type: str,
    orientation: str,
) -> Image.Image:
    """Create a per-orientation chord particle image with measured chords in black."""
    height, width = int(shape[0]), int(shape[1])
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    if chords.empty:
        return image
    for _, row in _chords_for_measure_orientation(chords, measure_type, orientation).iterrows():
        draw.line(
            (int(row["start_x"]), int(row["start_y"]), int(row["end_x"]), int(row["end_y"])),
            fill=(0, 0, 0),
            width=1,
        )
    return image


def make_binary_chord_overlay(
    airspace_mask: np.ndarray,
    chords: pd.DataFrame,
    *,
    measure_type: str,
    orientation: str,
) -> Image.Image:
    """Overlay measured chords in red on the binary airspace mask."""
    binary_rgb = np.stack([mask_to_uint8(airspace_mask)] * 3, axis=-1)
    image = Image.fromarray(binary_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for _, row in _chords_for_measure_orientation(chords, measure_type, orientation).iterrows():
        draw.line(
            (int(row["start_x"]), int(row["start_y"]), int(row["end_x"]), int(row["end_y"])),
            fill=(255, 0, 0),
            width=2,
        )
    return image


def make_qc_panel(
    rgb: np.ndarray,
    airspace_mask: np.ndarray,
    overlay: Image.Image,
    *,
    title: str,
    final_airspaces_rgb: np.ndarray | None = None,
    edge_exclusion_applied: bool = True,
) -> Image.Image:
    """Combine original, segmentation, component-area QC, and chord overlay into a QC panel."""
    original = Image.fromarray(rgb.astype(np.uint8, copy=False)).convert("RGB")
    binary = Image.fromarray(mask_to_uint8(airspace_mask)).convert("RGB")
    overlay_rgb = overlay.convert("RGB")
    final_airspaces = (
        Image.fromarray(final_airspaces_rgb.astype(np.uint8, copy=False)).convert("RGB")
        if final_airspaces_rgb is not None
        else None
    )

    target_height = min(512, max(original.height, 1))

    def resize_to_height(img: Image.Image) -> Image.Image:
        scale = target_height / img.height
        return img.resize((max(1, int(img.width * scale)), target_height), Image.Resampling.LANCZOS)

    panel_sources = [original, binary]
    labels = ["Original field", "Binary airspace mask"]
    if final_airspaces is not None:
        panel_sources.append(final_airspaces)
        labels.append(
            "Non-edge connected components (red)"
            if edge_exclusion_applied
            else "Configured final connected components (red)"
        )
    panel_sources.append(overlay_rgb)
    labels.append("Measured MLI chords")

    panels = [resize_to_height(img) for img in panel_sources]
    gap = 12
    header_height = 58 if final_airspaces is not None else 42
    label_height = 30
    width = sum(p.width for p in panels) + gap * (len(panels) + 1)
    height = header_height + label_height + target_height + gap
    out = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(out)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 18)
        label_font = ImageFont.truetype("Arial.ttf", 13)
        note_font = ImageFont.truetype("Arial.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        note_font = ImageFont.load_default()

    draw.text((gap, 10), title, fill=(16, 24, 40), font=title_font)
    if final_airspaces is not None:
        if edge_exclusion_applied:
            note = (
                "Area QC: red connected components are fully internal/non-edge; "
                "edge-touching components are excluded, so area is censored."
            )
        else:
            note = (
                "Area QC: red connected components are included by the configured final-area setting; "
                "edge-touching components are not excluded."
            )
        draw.text((gap, 34), note, fill=(52, 64, 84), font=note_font)
    x = gap
    y_label = header_height
    y_img = header_height + label_height
    for label, panel in zip(labels, panels):
        draw.text((x, y_label + 5), label, fill=(52, 64, 84), font=label_font)
        out.paste(panel, (x, y_img))
        x += panel.width + gap
    return out


def save_pil(image: Image.Image, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
