#!/usr/bin/env python3
"""
OpenCV CPU stress benchmark for small Linux boards.

The benchmark analyzes source pictures at their original resolution, or renders
synthetic scenes at common display resolutions when requested. It measures image
write/read, thresholding, morphology, contour extraction, and connected-
components analysis. It avoids GPU APIs so results are useful on Raspberry
Pi-like devices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
}

PROJECT_ROOT = Path(__file__).resolve().parent
PICTURE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Resolution:
    name: str
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass
class SampleResult:
    source: str
    source_image: str
    resolution: str
    iteration: int
    width: int
    height: int
    pixels: int
    cpu_cores: int
    opencv_threads: int
    input_read_ms: float
    render_ms: float
    benchmark_write_ms: float
    benchmark_read_ms: float
    write_ms: float
    read_ms: float
    grayscale_ms: float
    threshold_ms: float
    gaussian_blur_ms: float
    otsu_threshold_ms: float
    morphology_ms: float
    morphology_open_ms: float
    morphology_close_ms: float
    contours_ms: float
    connected_components_ms: float
    centroid_ms: float
    centroid_avg_us: float
    centroid_rate_per_second: float
    core_compute_ms: float
    stage_compute_ms: float
    stage_write_ms: float
    preview_write_ms: float
    output_write_ms: float
    processed_pipeline_ms: float
    total_ms: float
    image_size_bytes: int
    contours: int
    connected_components: int
    centroids: int
    shapes_rendered: int

    def to_row(self) -> dict[str, int | float | str]:
        return {
            "source": self.source,
            "source_image": self.source_image,
            "resolution": self.resolution,
            "iteration": self.iteration,
            "width": self.width,
            "height": self.height,
            "pixels": self.pixels,
            "cpu_cores": self.cpu_cores,
            "opencv_threads": self.opencv_threads,
            "input_read_ms": round(self.input_read_ms, 3),
            "render_ms": round(self.render_ms, 3),
            "benchmark_write_ms": round(self.benchmark_write_ms, 3),
            "benchmark_read_ms": round(self.benchmark_read_ms, 3),
            "write_ms": round(self.write_ms, 3),
            "read_ms": round(self.read_ms, 3),
            "grayscale_ms": round(self.grayscale_ms, 3),
            "threshold_ms": round(self.threshold_ms, 3),
            "gaussian_blur_ms": round(self.gaussian_blur_ms, 3),
            "otsu_threshold_ms": round(self.otsu_threshold_ms, 3),
            "morphology_ms": round(self.morphology_ms, 3),
            "morphology_open_ms": round(self.morphology_open_ms, 3),
            "morphology_close_ms": round(self.morphology_close_ms, 3),
            "contours_ms": round(self.contours_ms, 3),
            "connected_components_ms": round(self.connected_components_ms, 3),
            "centroid_ms": round(self.centroid_ms, 3),
            "centroid_avg_us": round(self.centroid_avg_us, 3),
            "centroid_rate_per_second": round(self.centroid_rate_per_second, 3),
            "core_compute_ms": round(self.core_compute_ms, 3),
            "stage_compute_ms": round(self.stage_compute_ms, 3),
            "stage_write_ms": round(self.stage_write_ms, 3),
            "preview_write_ms": round(self.preview_write_ms, 3),
            "output_write_ms": round(self.output_write_ms, 3),
            "processed_pipeline_ms": round(self.processed_pipeline_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "image_size_bytes": self.image_size_bytes,
            "contours": self.contours,
            "connected_components": self.connected_components,
            "centroids": self.centroids,
            "shapes_rendered": self.shapes_rendered,
        }


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


def parse_resolution(value: str) -> Resolution:
    key = value.lower()
    if key in RESOLUTIONS:
        width, height = RESOLUTIONS[key]
        return Resolution(key, width, height)

    separators = ("x", "X", "*")
    for separator in separators:
        if separator in value:
            raw_width, raw_height = value.split(separator, 1)
            try:
                width = int(raw_width.strip())
                height = int(raw_height.strip())
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid resolution '{value}', expected WIDTHxHEIGHT"
                ) from exc
            if width <= 0 or height <= 0:
                raise argparse.ArgumentTypeError("resolution dimensions must be positive")
            return Resolution(f"{width}x{height}", width, height)

    known = ", ".join(RESOLUTIONS)
    raise argparse.ArgumentTypeError(
        f"invalid resolution '{value}', use one of: {known}, or WIDTHxHEIGHT"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU/OpenCV stress benchmark using project pictures or rendered images."
    )
    parser.add_argument(
        "-r",
        "--resolutions",
        nargs="+",
        type=parse_resolution,
        default=[parse_resolution("720p"), parse_resolution("1080p")],
        help="synthetic fallback resolutions: 720p 1080p 2k 4k, or custom WIDTHxHEIGHT",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=3,
        help="iterations per resolution",
    )
    parser.add_argument(
        "--shapes",
        type=int,
        default=900,
        help="base number of shapes rendered at 1080p; scales by pixel count",
    )
    parser.add_argument(
        "--picture-dir",
        type=Path,
        default=Path("pictures"),
        help="directory with source pictures, relative paths are resolved from the project root",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        default=None,
        help="use one specific source image instead of scanning --picture-dir",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="ignore pictures and use the generated synthetic scene",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output"),
        help="directory for generated images and benchmark reports",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg", "bmp"),
        default="png",
        help="image format used for write/read I/O tests",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="OpenCV thread count. 0 keeps OpenCV default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260521,
        help="random seed for repeatable scenes",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="keep every generated image instead of deleting intermediate images",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="save one visual preview per resolution",
    )
    parser.add_argument(
        "--save-stages",
        action="store_true",
        help="save processed images: grayscale, binary, morphology, contours, and components",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run the benchmark under pyinstrument and write an HTML profile report",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="optional pyinstrument HTML report path, default is OUTPUT_DIR/profile.html",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional JSON report path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional CSV report path",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be greater than 0")
    if args.shapes <= 0:
        parser.error("--shapes must be greater than 0")
    if args.threads < 0:
        parser.error("--threads must be 0 or greater")
    if args.synthetic and args.source_image:
        parser.error("--synthetic cannot be used with --source-image")

    return args


def project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timed(operation: Callable[[], object]) -> tuple[object, float]:
    with Timer() as timer:
        result = operation()
    return result, timer.elapsed_ms


def shape_count_for_resolution(base_shapes: int, resolution: Resolution) -> int:
    base_pixels = RESOLUTIONS["1080p"][0] * RESOLUTIONS["1080p"][1]
    scaled = int(base_shapes * (resolution.pixels / base_pixels))
    return max(80, scaled)


def render_scene(
    resolution: Resolution,
    shape_count: int,
    rng: random.Random,
) -> np.ndarray:
    height, width = resolution.height, resolution.width
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    diagonal = (x_grid / max(width - 1, 1)) * 115.0 + (y_grid / max(height - 1, 1)) * 95.0
    wave = np.sin((x_grid + y_grid) / max(width / 17.0, 1.0)) * 35.0
    base = np.clip(diagonal + wave + 35.0, 0, 255).astype(np.uint8)
    image = cv2.merge(
        (
            base,
            np.roll(base, width // 23 or 1, axis=1),
            np.roll(base, height // 19 or 1, axis=0),
        )
    )

    min_side = min(width, height)
    for index in range(shape_count):
        color = (
            rng.randrange(25, 256),
            rng.randrange(25, 256),
            rng.randrange(25, 256),
        )
        thickness = -1 if index % 4 else rng.randrange(1, max(2, min_side // 260))
        mode = index % 5

        if mode == 0:
            center = (rng.randrange(width), rng.randrange(height))
            radius = rng.randrange(max(3, min_side // 160), max(5, min_side // 22))
            cv2.circle(image, center, radius, color, thickness, lineType=cv2.LINE_AA)
        elif mode == 1:
            x1 = rng.randrange(width)
            y1 = rng.randrange(height)
            x2 = min(width - 1, x1 + rng.randrange(max(8, width // 90), max(12, width // 9)))
            y2 = min(height - 1, y1 + rng.randrange(max(8, height // 90), max(12, height // 9)))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        elif mode == 2:
            points = np.array(
                [
                    [rng.randrange(width), rng.randrange(height)]
                    for _ in range(rng.randrange(3, 8))
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(image, [points], color, lineType=cv2.LINE_AA)
        elif mode == 3:
            center = (rng.randrange(width), rng.randrange(height))
            axes = (
                rng.randrange(max(4, width // 120), max(8, width // 18)),
                rng.randrange(max(4, height // 120), max(8, height // 18)),
            )
            angle = rng.randrange(0, 180)
            cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)
        else:
            pt1 = (rng.randrange(width), rng.randrange(height))
            pt2 = (rng.randrange(width), rng.randrange(height))
            line_thickness = rng.randrange(1, max(2, min_side // 180))
            cv2.line(image, pt1, pt2, color, line_thickness, lineType=cv2.LINE_AA)

    label = f"{resolution.name.upper()} OpenCV CPU benchmark"
    font_scale = max(0.7, min_side / 950.0)
    thickness = max(1, int(round(font_scale * 2)))
    cv2.putText(
        image,
        label,
        (max(12, width // 45), max(42, height // 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )
    return image


def discover_pictures(path: Path) -> list[Path]:
    resolved = project_path(path)
    if resolved.is_file():
        if resolved.suffix.lower() in PICTURE_EXTENSIONS:
            return [resolved]
        return []
    if not resolved.is_dir():
        return []
    return sorted(
        file
        for file in resolved.rglob("*")
        if file.is_file() and file.suffix.lower() in PICTURE_EXTENSIONS
    )


def resolution_from_image(name: str, image: np.ndarray) -> Resolution:
    height, width = image.shape[:2]
    return Resolution(name=f"{width}x{height}", width=width, height=height)


def load_picture_scene(source_image: Path) -> np.ndarray:
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read source image: {source_image}")
    return image


def encode_params(image_format: str) -> list[int]:
    if image_format == "jpg":
        return [cv2.IMWRITE_JPEG_QUALITY, 92]
    if image_format == "png":
        return [cv2.IMWRITE_PNG_COMPRESSION, 3]
    return []


def safe_stem(path: Path | None) -> str:
    if path is None:
        return "synthetic"
    return "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in path.stem)


def image_path(
    output_dir: Path,
    resolution: Resolution,
    iteration: int,
    image_format: str,
    source_image: Path | None,
) -> Path:
    stem = safe_stem(source_image)
    return output_dir / f"{stem}_{resolution.name}_{resolution.width}x{resolution.height}_{iteration}.{image_format}"


def output_stem(source_image: Path | None, resolution: Resolution, iteration: int) -> str:
    stem = safe_stem(source_image)
    return f"{stem}_{resolution.name}_{resolution.width}x{resolution.height}_{iteration}"


def write_image(path: Path, image: np.ndarray, image_format: str) -> int:
    ok = cv2.imwrite(str(path), image, encode_params(image_format))
    if not ok:
        raise RuntimeError(f"OpenCV failed to write image: {path}")
    return path.stat().st_size


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV failed to read image: {path}")
    return image


def preview_size(width: int, height: int, max_width: int = 1280, max_height: int = 720) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def colorize_components(labels: np.ndarray, component_count: int) -> np.ndarray:
    if component_count <= 1:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)

    hue = ((labels.astype(np.uint32) * 37) % 180).astype(np.uint8)
    saturation = np.full(labels.shape, 210, dtype=np.uint8)
    value = np.where(labels > 0, 255, 0).astype(np.uint8)
    hsv = cv2.merge((hue, saturation, value))
    colorized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    colorized[labels == 0] = (0, 0, 0)
    return colorized


def build_processed_images(
    loaded: np.ndarray,
    gray: np.ndarray,
    binary: np.ndarray,
    morphed: np.ndarray,
    contours: list[np.ndarray],
    labels: np.ndarray,
    component_count: int,
    resolution: Resolution,
) -> dict[str, np.ndarray]:
    contour_overlay = loaded.copy()
    cv2.drawContours(contour_overlay, contours, -1, (0, 255, 255), max(1, min(resolution.width, resolution.height) // 900))

    component_overlay = cv2.addWeighted(
        loaded,
        0.55,
        colorize_components(labels, component_count),
        0.45,
        0,
    )

    return {
        "gray": gray,
        "binary": binary,
        "morphology": morphed,
        "contours": contour_overlay,
        "components": component_overlay,
    }


def write_processed_images(
    output_dir: Path,
    source_image: Path | None,
    resolution: Resolution,
    iteration: int,
    outputs: dict[str, np.ndarray],
) -> list[str]:
    stage_dir = output_dir / "processed"
    ensure_output_dir(stage_dir)
    stem = output_stem(source_image, resolution, iteration)
    saved_paths: list[Path] = []

    for stage_name, image in outputs.items():
        path = stage_dir / f"{stem}_{stage_name}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved_paths.append(path)

    return [str(path.resolve()) for path in saved_paths]


def calculate_centroids(stats: np.ndarray) -> list[tuple[float, float]]:
    centroids: list[tuple[float, float]] = []
    for left, top, width, height, area in stats[1:]:
        if area <= 0:
            continue
        centroids.append((left + (width / 2.0), top + (height / 2.0)))
    return centroids


def run_sample(
    requested_resolution: Resolution,
    iteration: int,
    args: argparse.Namespace,
    rng: random.Random,
    source_image: Path | None,
) -> SampleResult:
    shape_count = 0 if source_image else shape_count_for_resolution(args.shapes, requested_resolution)
    total_start = time.perf_counter()

    if source_image:
        image, render_ms = timed(lambda: load_picture_scene(source_image))
        resolution = resolution_from_image(source_image.stem, image)
        source = "picture"
        source_name = str(source_image.relative_to(PROJECT_ROOT)) if source_image.is_relative_to(PROJECT_ROOT) else str(source_image)
    else:
        image, render_ms = timed(lambda: render_scene(requested_resolution, shape_count, rng))
        resolution = requested_resolution
        source = "synthetic"
        source_name = "generated"

    input_read_ms = render_ms
    path = image_path(args.output_dir, resolution, iteration, args.format, source_image)
    image_size_bytes, write_ms = timed(lambda: write_image(path, image, args.format))
    loaded, read_ms = timed(lambda: read_image(path))
    gray, grayscale_ms = timed(lambda: cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY))

    blurred, gaussian_blur_ms = timed(lambda: cv2.GaussianBlur(gray, (5, 5), 0))

    def otsu_threshold() -> np.ndarray:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    binary, otsu_threshold_ms = timed(otsu_threshold)
    threshold_ms = gaussian_blur_ms + otsu_threshold_ms

    kernel_size = max(3, 2 * math.ceil(min(resolution.width, resolution.height) / 720) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened, morphology_open_ms = timed(lambda: cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1))
    morphed, morphology_close_ms = timed(lambda: cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2))
    morphology_ms = morphology_open_ms + morphology_close_ms

    def contours() -> list[np.ndarray]:
        found, _hierarchy = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return found

    found_contours, contours_ms = timed(contours)

    def connected_components() -> tuple[int, np.ndarray, np.ndarray]:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(morphed, 8, cv2.CV_32S)
        return int(count), labels, stats

    component_result, connected_components_ms = timed(connected_components)
    component_count, labels, stats = component_result
    centroids, centroid_ms = timed(lambda: calculate_centroids(stats))
    centroid_count = len(centroids)
    centroid_avg_us = (centroid_ms * 1000.0 / centroid_count) if centroid_count else 0.0
    centroid_rate_per_second = (centroid_count / (centroid_ms / 1000.0)) if centroid_ms > 0 else 0.0

    preview_write_ms = 0.0

    if args.preview and iteration == 1:
        preview_path = args.output_dir / f"{output_stem(source_image, resolution, iteration)}_preview.jpg"
        preview = cv2.resize(loaded, preview_size(resolution.width, resolution.height))
        _ok, preview_write_ms = timed(
            lambda: cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
        )

    stage_compute_ms = 0.0
    stage_write_ms = 0.0

    if args.save_stages:
        processed_images, stage_compute_ms = timed(
            lambda: build_processed_images(
                loaded,
                gray,
                binary,
                morphed,
                found_contours,
                labels,
                component_count,
                resolution,
            )
        )
        _saved_paths, stage_write_ms = timed(
            lambda: write_processed_images(
                args.output_dir,
                source_image,
                resolution,
                iteration,
                processed_images,
            )
        )

    core_compute_ms = (
        grayscale_ms
        + threshold_ms
        + morphology_ms
        + contours_ms
        + connected_components_ms
        + centroid_ms
    )
    output_write_ms = stage_write_ms + preview_write_ms
    processed_pipeline_ms = core_compute_ms + stage_compute_ms + output_write_ms
    total_ms = (time.perf_counter() - total_start) * 1000.0

    if not args.keep_images:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return SampleResult(
        source=source,
        source_image=source_name,
        resolution=resolution.name,
        iteration=iteration,
        width=resolution.width,
        height=resolution.height,
        pixels=resolution.pixels,
        cpu_cores=os.cpu_count() or 0,
        opencv_threads=cv2.getNumThreads(),
        input_read_ms=input_read_ms,
        render_ms=render_ms,
        benchmark_write_ms=write_ms,
        benchmark_read_ms=read_ms,
        write_ms=write_ms,
        read_ms=read_ms,
        grayscale_ms=grayscale_ms,
        threshold_ms=threshold_ms,
        gaussian_blur_ms=gaussian_blur_ms,
        otsu_threshold_ms=otsu_threshold_ms,
        morphology_ms=morphology_ms,
        morphology_open_ms=morphology_open_ms,
        morphology_close_ms=morphology_close_ms,
        contours_ms=contours_ms,
        connected_components_ms=connected_components_ms,
        centroid_ms=centroid_ms,
        centroid_avg_us=centroid_avg_us,
        centroid_rate_per_second=centroid_rate_per_second,
        core_compute_ms=core_compute_ms,
        stage_compute_ms=stage_compute_ms,
        stage_write_ms=stage_write_ms,
        preview_write_ms=preview_write_ms,
        output_write_ms=output_write_ms,
        processed_pipeline_ms=processed_pipeline_ms,
        total_ms=total_ms,
        image_size_bytes=image_size_bytes,
        contours=len(found_contours),
        connected_components=component_count,
        centroids=centroid_count,
        shapes_rendered=shape_count,
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return statistics.fmean(values)


def summarize(samples: list[SampleResult]) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    by_resolution: dict[str, list[SampleResult]] = {}
    for sample in samples:
        by_resolution.setdefault(sample.resolution, []).append(sample)

    for resolution, group in by_resolution.items():
        pixels = group[0].pixels
        cpu_cores = group[0].cpu_cores
        opencv_threads = group[0].opencv_threads
        total_ms = mean(sample.total_ms for sample in group)
        cv_ms = mean(
            sample.grayscale_ms
            + sample.threshold_ms
            + sample.morphology_ms
            + sample.contours_ms
            + sample.connected_components_ms
            + sample.centroid_ms
            for sample in group
        )
        megapixels_per_second = (pixels / 1_000_000.0) / (total_ms / 1000.0)
        cv_megapixels_per_second = (pixels / 1_000_000.0) / (cv_ms / 1000.0)

        rows.append(
            {
                "resolution": resolution,
                "width": group[0].width,
                "height": group[0].height,
                "cpu_cores": cpu_cores,
                "opencv_threads": opencv_threads,
                "iterations": len(group),
                "avg_total_ms": round(total_ms, 3),
                "avg_input_read_ms": round(mean(sample.input_read_ms for sample in group), 3),
                "avg_render_ms": round(mean(sample.render_ms for sample in group), 3),
                "avg_benchmark_write_ms": round(mean(sample.benchmark_write_ms for sample in group), 3),
                "avg_benchmark_read_ms": round(mean(sample.benchmark_read_ms for sample in group), 3),
                "avg_write_ms": round(mean(sample.write_ms for sample in group), 3),
                "avg_read_ms": round(mean(sample.read_ms for sample in group), 3),
                "avg_grayscale_ms": round(mean(sample.grayscale_ms for sample in group), 3),
                "avg_threshold_ms": round(mean(sample.threshold_ms for sample in group), 3),
                "avg_gaussian_blur_ms": round(mean(sample.gaussian_blur_ms for sample in group), 3),
                "avg_otsu_threshold_ms": round(mean(sample.otsu_threshold_ms for sample in group), 3),
                "avg_morphology_ms": round(mean(sample.morphology_ms for sample in group), 3),
                "avg_morphology_open_ms": round(mean(sample.morphology_open_ms for sample in group), 3),
                "avg_morphology_close_ms": round(mean(sample.morphology_close_ms for sample in group), 3),
                "avg_contours_ms": round(mean(sample.contours_ms for sample in group), 3),
                "avg_connected_components_ms": round(
                    mean(sample.connected_components_ms for sample in group), 3
                ),
                "avg_centroid_ms": round(mean(sample.centroid_ms for sample in group), 3),
                "avg_centroid_avg_us": round(mean(sample.centroid_avg_us for sample in group), 3),
                "avg_centroid_rate_per_second": round(
                    mean(sample.centroid_rate_per_second for sample in group), 3
                ),
                "avg_core_compute_ms": round(mean(sample.core_compute_ms for sample in group), 3),
                "avg_stage_compute_ms": round(mean(sample.stage_compute_ms for sample in group), 3),
                "avg_stage_write_ms": round(mean(sample.stage_write_ms for sample in group), 3),
                "avg_preview_write_ms": round(mean(sample.preview_write_ms for sample in group), 3),
                "avg_output_write_ms": round(mean(sample.output_write_ms for sample in group), 3),
                "avg_processed_pipeline_ms": round(
                    mean(sample.processed_pipeline_ms for sample in group), 3
                ),
                "avg_image_size_kib": round(mean(sample.image_size_bytes for sample in group) / 1024.0, 1),
                "avg_contours": round(mean(sample.contours for sample in group), 1),
                "avg_connected_components": round(
                    mean(sample.connected_components for sample in group), 1
                ),
                "avg_centroids": round(mean(sample.centroids for sample in group), 1),
                "total_megapixels_per_second": round(megapixels_per_second, 3),
                "opencv_megapixels_per_second": round(cv_megapixels_per_second, 3),
            }
        )
    return rows


def machine_info() -> dict[str, str | int | None]:
    return {
        "python": sys.version.replace("\n", " "),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "opencv_threads": cv2.getNumThreads(),
    }


def write_csv(path: Path, samples: list[SampleResult]) -> None:
    ensure_output_dir(path.parent)
    rows = [sample.to_row() for sample in samples]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    samples: list[SampleResult],
    summary: list[dict[str, int | float | str]],
) -> None:
    ensure_output_dir(path.parent)
    payload = {
        "machine": machine_info(),
        "summary": summary,
        "samples": [sample.to_row() for sample in samples],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_sample_result(sample: SampleResult, title: str | None = None) -> None:
    if title:
        print()
        print(title)

    print(f"图片: {sample.source_image}")
    print(f"分辨率: {sample.width}x{sample.height}")
    print(f"CPU 核心/逻辑处理器: {sample.cpu_cores}")
    print(f"OpenCV threads: {sample.opencv_threads}")
    print()
    print(f"总耗时: {sample.total_ms:.3f} ms")
    print(f"输入读图: {sample.input_read_ms:.3f} ms")
    print(f"基准写图: {sample.benchmark_write_ms:.3f} ms")
    print(f"基准读图: {sample.benchmark_read_ms:.3f} ms")
    print(f"核心 OpenCV 计算: {sample.core_compute_ms:.3f} ms")
    print(f"输出处理图: {sample.output_write_ms:.3f} ms")
    print(f"完整处理流水线: {sample.processed_pipeline_ms:.3f} ms")
    print()
    print(f"Gaussian blur: {sample.gaussian_blur_ms:.3f} ms")
    print(f"Otsu 二值化: {sample.otsu_threshold_ms:.3f} ms")
    print(f"二值化总耗时: {sample.threshold_ms:.3f} ms")
    print(f"形态学总耗时: {sample.morphology_ms:.3f} ms")
    print(f"轮廓提取: {sample.contours_ms:.3f} ms")
    print(f"连通域分析: {sample.connected_components_ms:.3f} ms")
    print(f"质心计算: {sample.centroid_ms:.3f} ms")
    print()
    print(f"轮廓数: {sample.contours}")
    print(f"连通域数量: {sample.connected_components}")
    print(f"质心数量: {sample.centroids}")
    print(
        "OpenCV 吞吐: "
        f"{sample.pixels / 1_000_000.0 / (sample.core_compute_ms / 1000.0):.3f} MP/s"
    )


def print_summary(summary: list[dict[str, int | float | str]]) -> None:
    for row in summary:
        print()
        print("平均结果")
        print(f"分辨率: {row['width']}x{row['height']}")
        print(f"迭代次数: {row['iterations']}")
        print(f"CPU 核心/逻辑处理器: {row['cpu_cores']}")
        print(f"OpenCV threads: {row['opencv_threads']}")
        print()
        print(f"平均总耗时: {row['avg_total_ms']} ms")
        print(f"平均输入读图: {row['avg_input_read_ms']} ms")
        print(f"平均基准写图: {row['avg_benchmark_write_ms']} ms")
        print(f"平均基准读图: {row['avg_benchmark_read_ms']} ms")
        print(f"平均核心 OpenCV 计算: {row['avg_core_compute_ms']} ms")
        print(f"平均输出处理图: {row['avg_output_write_ms']} ms")
        print(f"平均完整处理流水线: {row['avg_processed_pipeline_ms']} ms")
        print()
        print(f"平均 Gaussian blur: {row['avg_gaussian_blur_ms']} ms")
        print(f"平均 Otsu 二值化: {row['avg_otsu_threshold_ms']} ms")
        print(f"平均二值化总耗时: {row['avg_threshold_ms']} ms")
        print(f"平均形态学总耗时: {row['avg_morphology_ms']} ms")
        print(f"平均轮廓提取: {row['avg_contours_ms']} ms")
        print(f"平均连通域分析: {row['avg_connected_components_ms']} ms")
        print(f"平均质心计算: {row['avg_centroid_ms']} ms")
        print()
        print(f"平均轮廓数: {row['avg_contours']}")
        print(f"平均连通域数量: {row['avg_connected_components']}")
        print(f"平均质心数量: {row['avg_centroids']}")
        print(f"平均 OpenCV 吞吐: {row['opencv_megapixels_per_second']} MP/s")


def run_benchmark(args: argparse.Namespace) -> int:
    ensure_output_dir(args.output_dir)

    if args.threads:
        cv2.setNumThreads(args.threads)

    samples: list[SampleResult] = []
    rng = random.Random(args.seed)
    source_images: list[Path] = []
    if args.source_image:
        source_images = discover_pictures(args.source_image)
        if not source_images:
            print(f"No supported source image found: {project_path(args.source_image)}", file=sys.stderr)
            return 2
    elif not args.synthetic:
        source_images = discover_pictures(args.picture_dir)

    print("OpenCV CPU stress benchmark")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"OpenCV threads: {cv2.getNumThreads()}")
    if source_images:
        print(f"Picture source: {len(source_images)} image(s)")
        print(f"First image: {source_images[0]}")
    elif args.synthetic:
        print("Picture source: synthetic scene forced by --synthetic")
    else:
        print(f"Picture source: no images found in {project_path(args.picture_dir)}, using synthetic scene")

    try:
        if source_images:
            for source_image in source_images:
                for iteration in range(1, args.iterations + 1):
                    sample = run_sample(args.resolutions[0], iteration, args, rng, source_image)
                    samples.append(sample)
                    print_sample_result(sample, f"测试结果 #{iteration}")
        else:
            for resolution in args.resolutions:
                shape_count = shape_count_for_resolution(args.shapes, resolution)
                for iteration in range(1, args.iterations + 1):
                    sample = run_sample(resolution, iteration, args, rng, None)
                    samples.append(sample)
                    print_sample_result(sample, f"测试结果 #{iteration}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    if not samples:
        print("No benchmark samples were produced.", file=sys.stderr)
        return 1

    summary = summarize(samples)
    print_summary(summary)

    csv_path = args.csv or args.output_dir / "samples.csv"
    json_path = args.json or args.output_dir / "report.json"
    write_csv(csv_path, samples)
    write_json(json_path, samples, summary)
    print(f"\nCSV report: {csv_path.resolve()}")
    print(f"JSON report: {json_path.resolve()}")
    return 0


def write_profile_report(args: argparse.Namespace, profiler: object) -> Path:
    profile_path = args.profile_output or args.output_dir / "profile.html"
    ensure_output_dir(profile_path.parent)
    html = profiler.output_html()
    profile_path.write_text(html, encoding="utf-8")
    return profile_path


def main() -> int:
    args = parse_args()

    if not args.profile:
        return run_benchmark(args)

    try:
        from pyinstrument import Profiler
    except ImportError:
        print(
            "pyinstrument is not installed. Install it with: "
            "python -m pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        return 2

    profiler = Profiler()
    profiler.start()
    try:
        exit_code = run_benchmark(args)
    finally:
        profiler.stop()

    profile_path = write_profile_report(args, profiler)
    print(f"Pyinstrument profile: {profile_path.resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
