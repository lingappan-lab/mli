"""Lingappan MLI Analyzer.

A robust, reproducible implementation of semi-automated mean linear intercept
(MLI) chord measurement for cropped lung histology fields.
"""

from .analysis import (
    DEFAULT_CALIBRATION_SOURCE,
    DEFAULT_PIXEL_HEIGHT_UM,
    DEFAULT_PIXEL_WIDTH_UM,
    UNRECORDED_CALIBRATION_SOURCE,
    AnalysisParams,
    AnalysisResult,
    analyze_image,
    process_files,
)

__all__ = [
    "DEFAULT_CALIBRATION_SOURCE",
    "DEFAULT_PIXEL_HEIGHT_UM",
    "DEFAULT_PIXEL_WIDTH_UM",
    "UNRECORDED_CALIBRATION_SOURCE",
    "AnalysisParams",
    "AnalysisResult",
    "analyze_image",
    "process_files",
]
__version__ = "0.1.0"
