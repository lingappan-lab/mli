from __future__ import annotations

import importlib

import pandas as pd
import pytest

gr = pytest.importorskip("gradio")


app = importlib.import_module("app")


def test_create_demo_smoke_returns_gradio_blocks():
    created = app.create_demo()

    assert isinstance(created, gr.Blocks)


def test_app_import_exposes_launch_appearance_and_launch_kwargs_payloads():
    assert app.demo is not None
    assert set(app.LAUNCH_APPEARANCE) == {"theme", "css", "js"}
    assert app.LAUNCH_APPEARANCE["theme"] is app.CLINICAL_THEME
    assert "app-header" in app.LAUNCH_APPEARANCE["css"]
    assert "enableMultipleDirectoryPickers" in app.LAUNCH_APPEARANCE["js"]
    assert app.LAUNCH_KWARGS == app.LAUNCH_APPEARANCE


def test_result_table_truncation_note_is_visible_when_preview_is_limited():
    frame = pd.DataFrame({"Slide": [f"slide-{index:03d}" for index in range(100)]})

    html = app.render_dataframe_table(
        frame,
        "Field summary",
        "The run completed, but no field summary rows were returned.",
        total_rows=125,
    )

    assert "Showing first 100 of 125 rows" in html
    assert "Download the results ZIP for the complete table" in html


def test_line_protocol_copy_explains_chords_plainly():
    copy = app.render_line_protocol_example()

    assert "one-pixel test lines" in copy
    assert "one continuous stretch of airspace" in copy
    assert "not wall-thickness measurements" in copy


def test_slide_separator_examples_explain_grouping_and_final_separator():
    copy = app.render_slide_separator_examples("_")

    assert "final occurrence" in copy
    assert "MouseA_0001.tif" in copy
    assert "MouseA_0002.tif" in copy
    assert "MouseA_left_lung_0003.tif" in copy
    assert "MouseA_left_lung" in copy
    assert "Files with the same Slide value are combined" in copy


def test_slide_separator_examples_adapt_to_hyphen_separator():
    copy = app.render_slide_separator_examples("-")

    assert "hyphen (-)" in copy
    assert "MouseA-0001.tif" in copy
    assert "MouseA-left_lung" not in copy
    assert "MouseA_left_lung-0003.tif" in copy
