from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("gradio")

import app
from lingappan_mli.ui import config as ui_config
from lingappan_mli.ui import validation as ui_validation


def _write_image(path: Path, size: tuple[int, int] = (4, 4)) -> Path:
    Image.new("RGB", size, color="white").save(path)
    return path


def _run_analysis_for_test(uploaded_files: list[str], **overrides):
    params = {
        "folder_uploads": None,
        "pixel_width_um": 0.57,
        "pixel_height_um": 0.57,
        "grid_strategy_choice": "count",
        "num_lines": 5,
        "line_spacing_um": 35.4,
        "grid_random_offset": False,
        "grid_random_seed": None,
        "orientation": "both",
        "threshold_method": "huang",
        "airspace_bright": True,
        "airspace_component_connectivity": 8,
        "min_chord_um": 0.0,
        "slide_roi_separator": "_",
        "field_selection_method": "Not recorded",
        "field_selection_notes": "",
        "field_exclusion_criteria": "None",
        "calibration_source": app.DEFAULT_CALIBRATION_SOURCE,
    }
    params.update(overrides)
    return app.run_analysis(
        uploaded_files,
        params["folder_uploads"],
        params["pixel_width_um"],
        params["pixel_height_um"],
        params["grid_strategy_choice"],
        params["num_lines"],
        params["line_spacing_um"],
        params["grid_random_offset"],
        params["grid_random_seed"],
        params["orientation"],
        params["threshold_method"],
        params["airspace_bright"],
        params["airspace_component_connectivity"],
        params["min_chord_um"],
        params["slide_roi_separator"],
        params["field_selection_method"],
        params["field_selection_notes"],
        params["field_exclusion_criteria"],
        params["calibration_source"],
    )


def test_export_root_rejects_unsafe_root_without_explicit_override(monkeypatch):
    monkeypatch.setenv(ui_config.EXPORT_ROOT_ENV, Path.cwd().anchor)
    monkeypatch.delenv(ui_config.ALLOW_UNSAFE_EXPORT_ROOT_ENV, raising=False)

    with pytest.raises(RuntimeError, match="Refusing unsafe export root"):
        ui_config.resolve_export_root()


def test_export_root_allows_unsafe_root_with_explicit_override(monkeypatch):
    unsafe_root = Path.cwd().anchor
    monkeypatch.setenv(ui_config.EXPORT_ROOT_ENV, unsafe_root)
    monkeypatch.setenv(ui_config.ALLOW_UNSAFE_EXPORT_ROOT_ENV, "1")

    assert ui_config.resolve_export_root() == Path(unsafe_root).resolve(strict=False)


def test_cleanup_stale_export_runs_only_removes_old_run_dirs(tmp_path):
    old_run = tmp_path / f"{ui_config.EXPORT_RUN_PREFIX}old"
    fresh_run = tmp_path / f"{ui_config.EXPORT_RUN_PREFIX}fresh"
    unrelated = tmp_path / "other_old_dir"
    old_run.mkdir()
    fresh_run.mkdir()
    unrelated.mkdir()

    now = time.time()
    old_mtime = now - 10_000
    os.utime(old_run, (old_mtime, old_mtime))
    os.utime(unrelated, (old_mtime, old_mtime))

    removed = ui_config.cleanup_stale_export_runs(tmp_path, ttl_seconds=60, now=now)

    assert removed == 1
    assert not old_run.exists()
    assert fresh_run.exists()
    assert unrelated.exists()


def test_remove_export_run_only_removes_safe_app_run_dirs(tmp_path):
    run_dir = tmp_path / f"{ui_config.EXPORT_RUN_PREFIX}123"
    unrelated = tmp_path / "other"
    run_dir.mkdir()
    unrelated.mkdir()

    assert ui_config.remove_export_run(run_dir, export_root=tmp_path) is True
    assert not run_dir.exists()
    assert ui_config.remove_export_run(unrelated, export_root=tmp_path) is False
    assert unrelated.exists()
    assert ui_config.remove_export_run(tmp_path, export_root=tmp_path) is False


def test_reset_analysis_workflow_removes_current_run_dir(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    run_dir = export_root / f"{ui_config.EXPORT_RUN_PREFIX}complete"
    run_dir.mkdir(parents=True)
    (run_dir / "result.zip").write_bytes(b"zip")
    monkeypatch.setattr(ui_config, "EXPORT_ROOT", export_root)

    result = app.reset_analysis_workflow(str(run_dir))

    assert len(result) == 15
    assert not run_dir.exists()
    assert result[-1] is None


def test_unload_cleanup_removes_session_export_dirs(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    run_dir = export_root / f"{ui_config.EXPORT_RUN_PREFIX}session"
    run_dir.mkdir(parents=True)
    (run_dir / "result.zip").write_bytes(b"zip")
    monkeypatch.setattr(ui_config, "EXPORT_ROOT", export_root)
    request = type("Request", (), {"session_hash": "session-1"})()

    app._register_session_export(run_dir, request)
    app.cleanup_session_exports(request)

    assert not run_dir.exists()
    assert "session-1" not in app._SESSION_EXPORT_RUNS


def test_upload_validation_reports_pixel_and_size_limits(tmp_path):
    image_path = _write_image(tmp_path / "field.png", size=(4, 4))

    errors = ui_validation.upload_validation_errors(
        [str(image_path)],
        max_upload_pixels=10,
        max_upload_file_bytes=1,
    )

    assert any("per-file limit" in error for error in errors)
    assert any("per-image limit" in error for error in errors)


def test_run_analysis_rejects_upload_before_process_files(tmp_path, monkeypatch):
    image_path = _write_image(tmp_path / "field.png", size=(4, 4))
    monkeypatch.setattr(
        app.ui_validation,
        "upload_validation_errors",
        lambda uploaded: ["field.png: image has 16 pixels, exceeding the per-image limit of 1 pixels."],
    )

    def fail_process_files(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("process_files should not be called for rejected uploads")

    monkeypatch.setattr(app, "process_files", fail_process_files)

    result = _run_analysis_for_test([str(image_path)])

    assert len(result) == 10
    assert "Upload limits need attention" in result[0]
    assert "per-image limit" in result[0]
    assert result[1]["visible"] is True
    assert result[2]["visible"] is False
    assert result[7]["visible"] is False
    assert result[8]["interactive"] is False
    assert result[9] is None


def test_run_analysis_logs_unexpected_exception_while_returning_error_html(tmp_path, monkeypatch, caplog):
    image_path = _write_image(tmp_path / "field.png", size=(4, 4))
    export_root = tmp_path / "exports"
    monkeypatch.setattr(ui_config, "EXPORT_ROOT", export_root)

    def fail_process_files(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "process_files", fail_process_files)

    with caplog.at_level(logging.ERROR, logger="app"):
        result = _run_analysis_for_test([str(image_path)])

    assert len(result) == 10
    assert "Analysis failed before outputs were packaged" in result[0]
    assert "RuntimeError: boom" in result[0]
    assert result[2]["visible"] is False
    assert result[7]["visible"] is False
    assert result[9] is None
    assert not any(export_root.iterdir())
    assert any("Unexpected error during Gradio analysis run" in record.message for record in caplog.records)


def test_run_analysis_success_returns_packaged_output_positions(tmp_path, monkeypatch):
    image_path = _write_image(tmp_path / "slideA_field1.png", size=(4, 4))
    export_root = tmp_path / "exports"
    captured = {}

    def fake_process_files(uploaded, output_dir, params, progress_callback):
        output_dir.mkdir(parents=True)
        (output_dir / "manifest.txt").write_text("ok")
        captured["uploaded"] = uploaded
        captured["params"] = params
        captured["progress_callback"] = progress_callback
        filename = Path(uploaded[0]).name
        return {
            "field_summary": pd.DataFrame(
                [
                    {
                        "filename": filename,
                        "slide_id": "slideA",
                        "field_id": "field1",
                        "mli_orientation_balanced_mean_um": 12.5,
                        "mli_chord_count": 7,
                        "mean_non_airspace_chord_um": 4.2,
                        "non_airspace_chord_count": 3,
                        "airspace_fraction": 0.4,
                        "non_edge_airspace_fraction": 0.3,
                        "non_edge_airspace_component_area_um2_mask": 21.0,
                        "calibration_is_default": False,
                    }
                ]
            ),
            "slide_summary": pd.DataFrame(
                [
                    {
                        "slide_id": "slideA",
                        "field_count": 1,
                        "mli_measured_field_count": 1,
                        "field_balanced_mean_mli_um": 12.5,
                        "field_sem_mli_um": 0.0,
                        "chord_pooled_mean_mli_um": 12.5,
                        "total_mli_chords": 7,
                        "non_airspace_measured_field_count": 1,
                        "field_balanced_mean_non_airspace_chord_um": 4.2,
                        "field_sd_non_airspace_chord_um": 0.0,
                        "field_sem_non_airspace_chord_um": 0.0,
                        "total_non_airspace_chords": 3,
                        "mean_airspace_fraction": 0.4,
                        "mean_non_edge_airspace_fraction": 0.3,
                        "mean_non_edge_airspace_component_area_um2_mask": 21.0,
                    }
                ]
            ),
            "processing_log": pd.DataFrame([{"filename": filename, "status": "ok", "message": ""}]),
            "preview_paths": [],
        }

    monkeypatch.setattr(ui_config, "EXPORT_ROOT", export_root)
    monkeypatch.setattr(app, "process_files", fake_process_files)

    result = _run_analysis_for_test(
        [str(image_path)],
        pixel_height_um=0.58,
        grid_random_offset=True,
        grid_random_seed=3,
        field_selection_notes="Selected by test",
    )

    assert len(result) == 10
    assert "Analysis complete and packaged" in result[0]
    assert result[1]["visible"] is False
    assert result[2]["visible"] is True
    assert "MLI (µm)" in result[2]["value"]
    assert "12.5" in result[2]["value"]
    assert result[4]["visible"] is True
    assert "Mean MLI (µm)" in result[4]["value"]
    assert result[5]["visible"] is True
    assert result[6]["visible"] is False
    assert Path(result[7]["value"]).is_file()
    assert result[8]["visible"] is True
    assert result[8]["interactive"] is True
    assert Path(result[9]).is_dir()
    assert Path(result[9]).parent == export_root
    assert captured["uploaded"] == [str(image_path)]
    assert captured["params"].pixel_width_um == 0.57
    assert captured["params"].pixel_height_um == 0.58
    assert captured["params"].grid_random_seed == 3


def test_field_display_uses_internal_row_key_for_duplicate_basename_logs():
    field_summary = pd.DataFrame(
        [
            {
                "filename": "SlideDup_0001.png",
                "slide_id": "SlideA",
                "field_id": "field1",
                "mli_orientation_balanced_mean_um": 10.0,
                "mli_chord_count": 5,
                "airspace_fraction": 0.25,
            },
            {
                "filename": "SlideDup_0001.png",
                "slide_id": "SlideB",
                "field_id": "field2",
                "mli_orientation_balanced_mean_um": 20.0,
                "mli_chord_count": 7,
                "airspace_fraction": 0.35,
            },
        ]
    )
    field_summary.attrs["_input_index"] = [1, 2]
    log_df = pd.DataFrame(
        [
            {"filename": "SlideDup_0001.png", "status": "ok", "message": ""},
            {
                "filename": "SlideDup_0001.png",
                "status": "warning",
                "message": "load warning",
            },
        ]
    )
    log_df.attrs["_input_index"] = [1, 2]

    display = app._field_display_source(
        field_summary,
        log_df,
        ["/first/SlideDup_0001.png", "/second/SlideDup_0001.png"],
        "_",
    )

    assert len(display) == 2
    assert display["slide_id"].tolist() == ["SlideA", "SlideB"]
    assert display["field_id"].tolist() == ["field1", "field2"]
    assert display["mli_orientation_balanced_mean_um"].tolist() == [10.0, 20.0]
    assert display["status"].tolist() == ["ok", "warning"]


def test_qc_preview_data_uri_only_reads_export_root(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    export_root.mkdir()
    inside = export_root / ui_config.EXPORT_RUN_PREFIX / "preview.png"
    inside.parent.mkdir()
    inside.write_bytes(b"preview")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    monkeypatch.setattr(ui_config, "EXPORT_ROOT", export_root)

    assert app._qc_preview_data_uri(inside).startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="outside export root"):
        app._qc_preview_data_uri(outside)


def test_qc_previews_expand_inline_without_opening_data_uri_links(tmp_path, monkeypatch):
    export_root = tmp_path / "exports"
    export_root.mkdir()
    preview = export_root / ui_config.EXPORT_RUN_PREFIX / "SlideA_0001" / "qc_panel.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"preview")
    monkeypatch.setattr(ui_config, "EXPORT_ROOT", export_root)

    html = app.render_qc_previews([preview])

    assert "data:image/png;base64," in html
    assert "<details class=\"qc-preview-item\">" in html
    assert "Click to expand inline" in html
    assert "href=\"data:image" not in html
    assert "target=\"_blank\"" not in html
