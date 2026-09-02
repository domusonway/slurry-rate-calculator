"""Production segmentation primitives for the Streamlit slurry-rate app.

The multi-segment implementation is derived from
``experiments/multisegment_benchmark.py``.  Region ids are ordered from dark to
bright illumination residuals so a saved parameter profile can be reused on
another image with the same acquisition setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    threshold_map: np.ndarray | None = None
    region_map: np.ndarray | None = None
    automatic_thresholds: list[float] | None = None
    applied_thresholds: list[float] | None = None


@dataclass
class MultiSegmentPreparation:
    normalized: np.ndarray
    region_map: np.ndarray
    automatic_thresholds: list[float]


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            output[labels == index] = 255
    return output


def postprocess(mask: np.ndarray, kernel_size: int = 5, min_area: int | None = None) -> np.ndarray:
    kernel_size = _odd(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    output = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    area = max(1, int(min_area if min_area is not None else max(64, mask.size // 12000)))
    return remove_small_components(output, area)


def illumination_normalize(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The lighting field only contains low-frequency information.  Estimating it
    # on a bounded working image avoids a very large Gaussian convolution on
    # camera-resolution inputs, then the field is mapped back without reducing
    # the resolution of the actual segmentation.
    height, width = gray.shape
    scale = min(1.0, 720.0 / max(height, width))
    if scale < 1.0:
        working = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        working = gray
    sigma = max(7.0, min(working.shape) / 18.0)
    field_working = cv2.GaussianBlur(working, (0, 0), sigmaX=sigma, sigmaY=sigma)
    field = (
        cv2.resize(field_working, (width, height), interpolation=cv2.INTER_LINEAR)
        if scale < 1.0
        else field_working
    )
    residual = gray.astype(np.float32) - field.astype(np.float32)
    low, high = np.percentile(residual, (1, 99))
    normalized = np.clip(
        (residual - low) * 255.0 / max(1.0, high - low), 0, 255
    ).astype(np.uint8)
    return normalized, residual


def _otsu_threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.uint8).reshape(-1, 1)
    if values.size == 0:
        return 127.0
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def _irregular_regions(residual: np.ndarray, max_regions: int = 6) -> np.ndarray:
    """Create coherent material-residual regions ordered from dark to bright."""
    max_regions = int(np.clip(max_regions, 2, 6))
    height, width = residual.shape
    scale = min(1.0, 360.0 / max(1, width))
    small = cv2.resize(residual, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(small, (0, 0), sigmaX=10, sigmaY=10)
    values = smooth.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.20)
    cv2.setRNGSeed(20260902)
    _, labels, centers = cv2.kmeans(
        values, max_regions, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    raw = labels.reshape(small.shape)
    order = np.argsort(centers[:, 0])
    ordered = np.zeros_like(raw, np.uint8)
    for new_id, old_id in enumerate(order):
        ordered[raw == old_id] = new_id
    ordered = cv2.medianBlur(ordered, 31)
    regions = cv2.resize(ordered, (width, height), interpolation=cv2.INTER_NEAREST)

    min_pixels = int(height * width * 0.025)
    for region_id in range(max_regions):
        selector = regions == region_id
        area = int(np.count_nonzero(selector))
        if not 0 < area < min_pixels:
            continue
        mean = float(residual[selector].mean())
        choices: list[tuple[float, int]] = []
        for other in range(max_regions):
            other_selector = regions == other
            if other == region_id or not np.any(other_selector):
                continue
            choices.append((abs(mean - float(residual[other_selector].mean())), other))
        if choices:
            regions[selector] = min(choices)[1]

    compact = np.zeros_like(regions, np.uint8)
    for new_id, old_id in enumerate(sorted(np.unique(regions).tolist())):
        compact[regions == old_id] = new_id
    return compact


def prepare_multisegment(image: np.ndarray, max_regions: int = 6) -> MultiSegmentPreparation:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized, residual = illumination_normalize(gray)
    regions = _irregular_regions(residual, max_regions=max_regions)
    thresholds = [
        _otsu_threshold(normalized[regions == region_id])
        for region_id in np.unique(regions)
    ]
    return MultiSegmentPreparation(normalized, regions, thresholds)


def segment_multisegment(
    image: np.ndarray,
    max_regions: int = 6,
    foreground_dark: bool = True,
    region_values: Mapping[int | str, float] | None = None,
    application_mode: str = "fixed",
    kernel_size: int = 5,
    min_area: int | None = None,
) -> SegmentationResult:
    preparation = prepare_multisegment(image, max_regions=max_regions)
    normalized = preparation.normalized
    regions = preparation.region_map
    mask = np.zeros_like(normalized)
    threshold_map = np.zeros_like(normalized, np.float32)
    applied: list[float] = []
    values = region_values or {}

    for list_index, region_id in enumerate(np.unique(regions)):
        automatic = preparation.automatic_thresholds[list_index]
        numeric_id = int(region_id)
        has_configured_value = numeric_id in values or str(numeric_id) in values
        configured = float(values.get(numeric_id, values.get(str(numeric_id), 0.0)))
        if application_mode == "relative":
            threshold = automatic + configured
        elif has_configured_value:
            threshold = configured
        else:
            threshold = automatic
        threshold = float(np.clip(threshold, 0, 255))
        selector = regions == region_id
        threshold_map[selector] = threshold
        if foreground_dark:
            mask[selector & (normalized <= threshold)] = 255
        else:
            mask[selector & (normalized >= threshold)] = 255
        applied.append(threshold)

    return SegmentationResult(
        postprocess(mask, kernel_size=kernel_size, min_area=min_area),
        threshold_map=threshold_map,
        region_map=regions,
        automatic_thresholds=preparation.automatic_thresholds,
        applied_thresholds=applied,
    )


def segment_image(
    image: np.ndarray,
    method: str,
    *,
    foreground_dark: bool = True,
    global_threshold: int = 160,
    adaptive_block: int = 51,
    adaptive_c: int = 3,
    max_regions: int = 6,
    region_values: Mapping[int | str, float] | None = None,
    application_mode: str = "fixed",
    kernel_size: int = 5,
    min_area: int | None = None,
) -> SegmentationResult:
    if image is None or image.size == 0:
        raise ValueError("图像为空")
    if method == "multisegment":
        return segment_multisegment(
            image,
            max_regions=max_regions,
            foreground_dark=foreground_dark,
            region_values=region_values,
            application_mode=application_mode,
            kernel_size=kernel_size,
            min_area=min_area,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_type = cv2.THRESH_BINARY_INV if foreground_dark else cv2.THRESH_BINARY
    automatic_thresholds: list[float] | None = None
    if method == "global":
        _, mask = cv2.threshold(gray, int(global_threshold), 255, threshold_type)
        applied = [float(global_threshold)]
    elif method == "otsu":
        threshold, mask = cv2.threshold(gray, 0, 255, threshold_type + cv2.THRESH_OTSU)
        automatic_thresholds = [float(threshold)]
        applied = [float(threshold)]
    elif method == "adaptive":
        block = _odd(adaptive_block)
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block,
            int(adaptive_c),
        )
        applied = None
    else:
        raise ValueError(f"未知分割算法：{method}")

    return SegmentationResult(
        postprocess(mask, kernel_size=kernel_size, min_area=min_area),
        automatic_thresholds=automatic_thresholds,
        applied_thresholds=applied,
    )


def coverage_percent(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask > 0) * 100.0)


def colorize_regions(regions: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [255, 170, 64],
            [120, 200, 70],
            [55, 160, 220],
            [220, 85, 170],
            [205, 190, 75],
            [100, 95, 230],
        ],
        dtype=np.uint8,
    )
    return palette[regions % len(palette)]
