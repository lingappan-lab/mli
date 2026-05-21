from __future__ import annotations

import json
import zipfile

import numpy as np
from PIL import Image

from lingappan_mli.analysis import AnalysisParams
from lingappan_mli.cli import build_parser, main


def _write_tiny_synthetic_field(path):
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :] = [70, 70, 90]
    arr[8:24, 5:13] = [240, 240, 240]
    arr[8:24, 19:27] = [240, 240, 240]
    Image.fromarray(arr).save(path)


def test_cli_random_grid_phase_defaults_to_enabled_and_auto_seeded():
    args = build_parser().parse_args(["field.tif"])

    assert args.random_grid_offset is True
    assert args.grid_random_seed is None

    params = AnalysisParams(
        grid_random_offset=args.random_grid_offset,
        grid_random_seed=args.grid_random_seed,
    ).validated()
    assert params.grid_random_offset is True
    assert params.grid_random_seed is not None
    assert params.grid_random_seed >= 0


def test_cli_random_grid_phase_can_be_disabled():
    args = build_parser().parse_args(["field.tif", "--no-random-grid-offset"])

    assert args.random_grid_offset is False

    params = AnalysisParams(
        grid_random_offset=args.random_grid_offset,
        grid_random_seed=args.grid_random_seed,
    ).validated()
    assert params.grid_random_offset is False
    assert params.grid_random_seed is None


def test_cli_random_grid_seed_zero_is_preserved():
    args = build_parser().parse_args(["field.tif", "--grid-random-seed", "0"])

    params = AnalysisParams(
        grid_random_offset=args.random_grid_offset,
        grid_random_seed=args.grid_random_seed,
    ).validated()
    assert params.grid_random_offset is True
    assert params.grid_random_seed == 0


def test_cli_help_uses_single_measurement_path():
    help_text = " ".join(build_parser().format_help().split())

    assert "--measurement-mode" not in help_text
    assert "imagej_overlay" not in help_text


def test_cli_main_processes_tiny_synthetic_image(tmp_path, capsys):
    image_path = tmp_path / "SlideCLI_0001.png"
    output_dir = tmp_path / "results"
    _write_tiny_synthetic_field(image_path)

    exit_code = main(
        [
            str(image_path),
            "--output",
            str(output_dir),
            "--pixel-width-um",
            "1",
            "--pixel-height-um",
            "1",
            "--num-lines",
            "4",
            "--threshold-method",
            "otsu",
            "--grid-random-seed",
            "123",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Processed fields: 1" in captured.out
    assert "Errors: 0" in captured.out
    assert captured.err == ""
    assert (output_dir / "lingappan_mli_results.xlsx").exists()
    assert (output_dir / "all_chords.csv").exists()
    assert (output_dir / "SlideCLI_0001" / "qc_panel.png").exists()
    with open(output_dir / "parameters.json", encoding="utf-8") as handle:
        parameters = json.load(handle)
    assert parameters["grid_random_offset"] is True
    assert parameters["grid_random_seed"] == 123


def test_cli_main_zip_option_creates_archive(tmp_path, capsys):
    image_path = tmp_path / "SlideZip_0001.png"
    output_dir = tmp_path / "zip_results"
    _write_tiny_synthetic_field(image_path)

    exit_code = main(
        [
            str(image_path),
            "--output",
            str(output_dir),
            "--pixel-width-um",
            "1",
            "--pixel-height-um",
            "1",
            "--num-lines",
            "3",
            "--threshold-method",
            "otsu",
            "--grid-random-seed",
            "123",
            "--zip",
        ]
    )

    captured = capsys.readouterr()
    archive_path = output_dir.with_suffix(".zip")
    assert exit_code == 0
    assert f"ZIP archive: {archive_path}" in captured.out
    assert captured.err == ""
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "lingappan_mli_results.xlsx" in names
    assert "parameters.json" in names


def test_cli_main_reports_no_supported_images_without_traceback(tmp_path, capsys):
    empty_input = tmp_path / "empty_inputs"
    empty_input.mkdir()

    exit_code = main([str(empty_input), "--output", str(tmp_path / "results")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "No supported images found" in captured.err
    assert "Traceback" not in captured.err


def test_cli_main_reports_parameter_errors_without_traceback(tmp_path, capsys):
    image_path = tmp_path / "SlideBadParams_0001.png"
    _write_tiny_synthetic_field(image_path)

    exit_code = main([str(image_path), "--pixel-width-um", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Pixel width/height must be positive" in captured.err
    assert "Traceback" not in captured.err


def test_cli_main_reports_preflight_errors_without_traceback(tmp_path, capsys):
    image_path = tmp_path / "SlideTooLarge_0001.png"
    output_dir = tmp_path / "results"
    _write_tiny_synthetic_field(image_path)

    exit_code = main(
        [
            str(image_path),
            "--output",
            str(output_dir),
            "--max-image-pixels",
            "10",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Image preflight failed" in captured.err
    assert "per-image limit" in captured.err
    assert "Traceback" not in captured.err
    assert not output_dir.exists()


def test_cli_main_reports_discovery_io_errors_without_traceback(tmp_path, capsys, monkeypatch):
    def raise_os_error(_paths):
        raise OSError("permission denied")

    monkeypatch.setattr("lingappan_mli.cli.discover_images", raise_os_error)

    exit_code = main([str(tmp_path), "--output", str(tmp_path / "results")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "could not discover input images" in captured.err
    assert "permission denied" in captured.err
    assert "Traceback" not in captured.err
