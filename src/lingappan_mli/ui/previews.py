"""Preview image renderers for the Gradio UI."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from lingappan_mli.grid import generate_grid
from lingappan_mli.measure import measure_phase_chords
from lingappan_mli.thresholding import threshold_airspace
from lingappan_mli.ui.config import (
    EXAMPLE_IMAGE_DIMS,
    EXAMPLE_IMAGE_PREVIEW,
    EXAMPLE_LINE_SPACING_UM,
    EXAMPLE_PIXEL_HEIGHT_UM,
    EXAMPLE_PIXEL_WIDTH_UM,
)
from lingappan_mli.ui.validation import (
    airspace_bright_value,
    optional_int,
    safe_float,
    safe_int,
    strategy_value,
)


def _draw_double_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill, width: int = 3) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    if sy == ey:
        arrow = 8
        draw.polygon([(sx, sy), (sx + arrow, sy - arrow // 2), (sx + arrow, sy + arrow // 2)], fill=fill)
        draw.polygon([(ex, ey), (ex - arrow, ey - arrow // 2), (ex - arrow, ey + arrow // 2)], fill=fill)
    elif sx == ex:
        arrow = 8
        draw.polygon([(sx, sy), (sx - arrow // 2, sy + arrow), (sx + arrow // 2, sy + arrow)], fill=fill)
        draw.polygon([(ex, ey), (ex - arrow // 2, ey - arrow), (ex + arrow // 2, ey - arrow)], fill=fill)


def render_calibration_preview(pixel_width_um: float, pixel_height_um: float):
    if EXAMPLE_IMAGE_PREVIEW is None:
        return None

    display = EXAMPLE_IMAGE_PREVIEW.convert("RGB")
    display_width, display_height = display.size
    original_width, original_height = EXAMPLE_IMAGE_DIMS or display.size
    pixel_width = max(safe_float(pixel_width_um, EXAMPLE_PIXEL_WIDTH_UM), 0.0001)
    pixel_height = max(safe_float(pixel_height_um, EXAMPLE_PIXEL_HEIGHT_UM), 0.0001)
    field_width_um = original_width * pixel_width
    field_height_um = original_height * pixel_height

    margin_left = 88
    margin_top = 72
    margin_right = 18
    margin_bottom = 86
    canvas = PILImage.new(
        "RGBA",
        (display_width + margin_left + margin_right, display_height + margin_top + margin_bottom),
        (255, 255, 255, 255),
    )
    canvas.paste(display.convert("RGBA"), (margin_left, margin_top))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("Arial.ttf", 16)
        label_font = ImageFont.truetype("Arial.ttf", 13)
        small_font = ImageFont.truetype("Arial.ttf", 12)
    except OSError:
        title_font = label_font = small_font = ImageFont.load_default()

    ink = (16, 24, 40, 255)
    muted = (52, 64, 84, 255)
    primary = (25, 118, 132, 255)
    accent = (245, 184, 0, 255)
    x0, y0 = margin_left, margin_top
    x1, y1 = margin_left + display_width, margin_top + display_height

    draw.text(
        (12, 14),
        "Calibration example",
        fill=ink,
        font=title_font,
    )
    draw.text(
        (12, 40),
        f"Pixel size: {pixel_width:g} × {pixel_height:g} µm/px · Original image: {original_width} × {original_height} px",
        fill=muted,
        font=label_font,
    )
    draw.rectangle((x0, y0, x1, y1), outline=(25, 118, 132, 210), width=2)

    h_arrow_y = y1 + 30
    _draw_double_arrow(draw, (x0, h_arrow_y), (x1, h_arrow_y), primary, width=3)
    h_label = f"width: {original_width:,} px × {pixel_width:g} µm/px = {field_width_um:,.1f} µm"
    h_bbox = draw.textbbox((0, 0), h_label, font=label_font)
    draw.text((x0 + (display_width - (h_bbox[2] - h_bbox[0])) / 2, h_arrow_y + 10), h_label, fill=ink, font=label_font)

    v_arrow_x = x0 - 34
    _draw_double_arrow(draw, (v_arrow_x, y0), (v_arrow_x, y1), primary, width=3)
    v_label = f"height: {original_height:,} px × {pixel_height:g} µm/px = {field_height_um:,.1f} µm"
    label_img = PILImage.new("RGBA", (display_height, 24), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 4), v_label, fill=ink, font=label_font)
    rotated = label_img.rotate(90, expand=True)
    canvas.alpha_composite(rotated, (8, y0 + max(0, (display_height - rotated.height) // 2)))

    scale_bar_um = 100
    scale_display_px = max(24, min(180, int((scale_bar_um / pixel_width) * (display_width / max(original_width, 1)))))
    bar_x1 = x1 - 24
    bar_x0 = bar_x1 - scale_display_px
    bar_y = y1 - 24
    draw.line((bar_x0, bar_y, bar_x1, bar_y), fill=accent, width=5)
    draw.line((bar_x0, bar_y - 8, bar_x0, bar_y + 8), fill=accent, width=3)
    draw.line((bar_x1, bar_y - 8, bar_x1, bar_y + 8), fill=accent, width=3)
    draw.rectangle((bar_x0 - 8, bar_y - 35, bar_x1 + 8, bar_y - 13), fill=(255, 255, 255, 210))
    draw.text((bar_x0, bar_y - 33), f"{scale_bar_um} µm", fill=ink, font=small_font)

    return canvas.convert("RGB")


def render_filter_preview(threshold_method: str, airspace_bright):
    if EXAMPLE_IMAGE_PREVIEW is None:
        return None

    airspace_bright_bool = airspace_bright_value(airspace_bright)
    display = EXAMPLE_IMAGE_PREVIEW.convert("RGB")
    rgb = np.asarray(display, dtype=np.uint8)
    _, airspace_mask, threshold = threshold_airspace(
        rgb,
        method=(threshold_method or "huang").lower(),
        airspace_bright=airspace_bright_bool,
    )
    base = display.convert("RGBA")
    mask_rgba = np.zeros((base.height, base.width, 4), dtype=np.uint8)
    mask_rgba[airspace_mask] = (42, 111, 219, 120)
    overlay = PILImage.fromarray(mask_rgba)
    composed = PILImage.alpha_composite(base, overlay)

    legend_height = 42
    canvas = PILImage.new("RGBA", (base.width, base.height + legend_height), (255, 255, 255, 255))
    canvas.paste(composed, (0, legend_height))
    legend = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    threshold_label = f"{(threshold_method or 'huang').title()} threshold: {threshold}"
    polarity_label = "bright airspaces" if airspace_bright_bool else "dark airspaces"
    legend.text((12, 12), f"Example filter preview · {threshold_label} · {polarity_label}", fill=(16, 24, 40, 255), font=font)
    legend.rectangle((base.width - 74, 13, base.width - 42, 29), fill=(42, 111, 219, 120), outline=(42, 111, 219, 220))
    legend.text((base.width - 36, 10), "airspace", fill=(52, 64, 84, 255), font=font)
    return canvas.convert("RGB")


def render_line_protocol_preview(
    pixel_width_um: float,
    pixel_height_um: float,
    grid_strategy_choice: str,
    num_lines: int,
    line_spacing_um: float,
    orientation: str,
    min_chord_um: float,
    threshold_method: str = "huang",
    airspace_bright: bool = True,
    grid_random_offset: bool = False,
    grid_random_seed=None,
):
    if EXAMPLE_IMAGE_PREVIEW is None:
        return None

    display_rgb = EXAMPLE_IMAGE_PREVIEW.convert("RGB")
    display = display_rgb.convert("RGBA")
    display_width, display_height = display.size
    pixel_width = safe_float(pixel_width_um, EXAMPLE_PIXEL_WIDTH_UM)
    pixel_height = safe_float(pixel_height_um, EXAMPLE_PIXEL_HEIGHT_UM)
    if EXAMPLE_IMAGE_DIMS is not None:
        original_width, original_height = EXAMPLE_IMAGE_DIMS
        pixel_width *= original_width / max(display_width, 1)
        pixel_height *= original_height / max(display_height, 1)
    pixel_width = max(pixel_width, 0.0001)
    pixel_height = max(pixel_height, 0.0001)

    _, airspace_mask, _ = threshold_airspace(
        np.asarray(display_rgb, dtype=np.uint8),
        method=(threshold_method or "huang").lower(),
        airspace_bright=airspace_bright_value(airspace_bright),
    )
    preview_seed = optional_int(grid_random_seed)
    preview_grid_seed = 0 if bool(grid_random_offset) and preview_seed is None else preview_seed
    lines = generate_grid(
        (display_height, display_width),
        orientation=(orientation or "both").lower(),
        strategy=strategy_value(grid_strategy_choice),
        num_lines=max(safe_int(num_lines, 15), 1),
        line_spacing_um=max(safe_float(line_spacing_um, EXAMPLE_LINE_SPACING_UM), 0.01),
        pixel_width_um=pixel_width,
        pixel_height_um=pixel_height,
        random_offset=bool(grid_random_offset),
        random_seed=preview_grid_seed,
    )

    min_chord = max(safe_float(min_chord_um, 0.0), 0.0)
    airspace_chords = measure_phase_chords(
        airspace_mask,
        lines,
        phase="airspace",
        pixel_width_um=pixel_width,
        pixel_height_um=pixel_height,
        exclude_edge_touching=True,
        min_chord_um=min_chord,
    )
    tissue_chords = measure_phase_chords(
        airspace_mask,
        lines,
        phase="tissue",
        pixel_width_um=pixel_width,
        pixel_height_um=pixel_height,
        exclude_edge_touching=True,
        min_chord_um=min_chord,
    )

    overlay = PILImage.new("RGBA", display.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grid_color = (42, 111, 219, 145)
    airspace_color = (245, 184, 0, 235)
    tissue_color = (190, 86, 255, 220)
    for line in lines:
        draw.line((line.x1, line.y1, line.x2, line.y2), fill=grid_color, width=1)
    for chords, color, width in ((airspace_chords, airspace_color, 3), (tissue_chords, tissue_color, 2)):
        if chords.empty:
            continue
        for _, row in chords.iterrows():
            draw.line(
                (int(row["start_x"]), int(row["start_y"]), int(row["end_x"]), int(row["end_y"])),
                fill=color,
                width=width,
            )

    composed = PILImage.alpha_composite(display, overlay)
    legend_height = 46
    canvas = PILImage.new("RGBA", (display_width, display_height + legend_height), (255, 255, 255, 255))
    canvas.paste(composed, (0, legend_height))
    legend = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    legend.text(
        (12, 8),
        f"Filter applied · MLI airspace chords: {len(airspace_chords)} · non-airspace chords: {len(tissue_chords)}",
        fill=(16, 24, 40, 255),
        font=font,
    )
    legend.line((12, 32, 42, 32), fill=grid_color, width=2)
    legend.text((48, 24), "test lines", fill=(52, 64, 84, 255), font=font)
    legend.line((136, 32, 166, 32), fill=airspace_color, width=3)
    legend.text((172, 24), "airspace", fill=(52, 64, 84, 255), font=font)
    legend.line((260, 32, 290, 32), fill=tissue_color, width=3)
    legend.text((296, 24), "non-airspace", fill=(52, 64, 84, 255), font=font)
    return canvas.convert("RGB")
