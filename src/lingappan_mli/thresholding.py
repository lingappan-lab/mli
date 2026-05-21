"""Thresholding routines.

The default Huang threshold follows the semi-automated Fiji/ImageJ workflow
used by Crowley et al. (BMC Pulmonary Medicine, 2019) for MLI chord isolation.
"""

from __future__ import annotations

import math

import numpy as np


def rgb_to_gray_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to 8-bit grayscale using luminance weights."""
    if rgb.ndim == 2:
        gray = rgb
    else:
        rgb_float = rgb.astype(np.float32, copy=False)
        gray = 0.299 * rgb_float[..., 0] + 0.587 * rgb_float[..., 1] + 0.114 * rgb_float[..., 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def huang_threshold(gray: np.ndarray) -> int:
    """Compute Huang's fuzzy threshold for an 8-bit grayscale image.

    This implementation is intentionally dependency-light and handles constant
    or nearly constant images gracefully.
    """
    gray = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    non_zero = np.nonzero(hist)[0]
    if non_zero.size == 0:
        return 0

    first_bin = int(non_zero[0])
    last_bin = int(non_zero[-1])
    if first_bin >= last_bin:
        return first_bin

    indices = np.arange(256, dtype=np.float64)
    num_pix_cumsum = np.cumsum(hist)
    sum_pix_cumsum = np.cumsum(indices * hist)
    mu_0 = sum_pix_cumsum / np.where(num_pix_cumsum == 0, 1, num_pix_cumsum)

    num_pix_cumsum_rev = np.cumsum(hist[::-1])[::-1]
    sum_pix_cumsum_rev = np.cumsum((indices[::-1]) * hist[::-1])[::-1]
    mu_1 = sum_pix_cumsum_rev / np.where(num_pix_cumsum_rev == 0, 1, num_pix_cumsum_rev)

    term = 1.0 / max(1, last_bin - first_bin)
    best_threshold = first_bin
    min_entropy = float("inf")

    for threshold in range(first_bin, last_bin + 1):
        entropy = 0.0
        for ih in range(first_bin, threshold + 1):
            if hist[ih] == 0:
                continue
            mu_x = 1.0 / (1.0 + term * abs(ih - mu_0[threshold]))
            if 1e-6 < mu_x < 1.0 - 1e-6:
                entropy -= hist[ih] * (mu_x * math.log(mu_x) + (1.0 - mu_x) * math.log(1.0 - mu_x))

        for ih in range(threshold + 1, last_bin + 1):
            if hist[ih] == 0:
                continue
            mu_x = 1.0 / (1.0 + term * abs(ih - mu_1[threshold]))
            if 1e-6 < mu_x < 1.0 - 1e-6:
                entropy -= hist[ih] * (mu_x * math.log(mu_x) + (1.0 - mu_x) * math.log(1.0 - mu_x))

        if entropy < min_entropy:
            min_entropy = entropy
            best_threshold = threshold

    return int(best_threshold)


def otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu's threshold for optional QC comparisons."""
    gray = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0

    bins = np.arange(256, dtype=np.float64)
    sum_total = float((bins * hist).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    max_between = -1.0
    threshold = 0

    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = i
    return int(threshold)


def threshold_airspace(
    rgb: np.ndarray,
    method: str = "huang",
    airspace_bright: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return grayscale image, airspace mask, and threshold value.

    Parameters
    ----------
    rgb:
        RGB uint8 image.
    method:
        ``"huang"`` by default, or ``"otsu"`` for optional QC.
    airspace_bright:
        If true, pixels above the threshold are airspace. This is the expected
        behavior for bright airspaces in H&E lung fields.
    """
    gray = rgb_to_gray_uint8(rgb)
    method_norm = method.lower().strip()
    if method_norm == "huang":
        threshold = huang_threshold(gray)
    elif method_norm == "otsu":
        threshold = otsu_threshold(gray)
    else:
        raise ValueError(f"Unsupported threshold method: {method!r}")

    if airspace_bright:
        airspace = gray > threshold
    else:
        airspace = gray <= threshold
    return gray, airspace.astype(bool, copy=False), int(threshold)
