#!/usr/bin/env python3
"""
Process a video with the existing LED localisation code, write one CSV row per
sampled frame, store each frame homography, then optionally render an annotated
video with the tracked path projected back into each frame.

Expected to live next to:
  measure_LED.py
  Measure.py
  Detect.py

Example:
  python track_led_video.py input.mp4 -out results_video.csv -annotated tracked.mp4

Two-step usage:
  python track_led_video.py input.mp4 -out results_video.csv --measure-only
  python track_led_video.py input.mp4 -csv results_video.csv -annotated tracked.mp4 --draw-only
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from measure_LED import Measure, WORLD_TL_COORDS, localise_led


CSV_FIELDS = [
    "frame",
    "time_s",
    "x",
    "y",
    "ok",
    "error",
    "n_calibration_markers_0_9",
    "led_marker_ids",
    "h00",
    "h01",
    "h02",
    "h10",
    "h11",
    "h12",
    "h20",
    "h21",
    "h22",
]


def build_measure() -> Measure:
    return Measure(WORLD_TL_COORDS, "tl")


def flatten_homography(H: np.ndarray | None) -> list[str]:
    if H is None:
        return [""] * 9
    H = np.asarray(H, dtype=float)
    return [f"{v:.12g}" for v in H.reshape(-1)]


def parse_homography(row: dict[str, str]) -> np.ndarray | None:
    keys = [f"h{i}{j}" for i in range(3) for j in range(3)]
    values = []
    for key in keys:
        text = row.get(key, "")
        if text is None or text == "":
            return None
        values.append(float(text))
    return np.array(values, dtype=np.float64).reshape(3, 3)


def measure_frame(
    frame_bgr: np.ndarray,
    min_calibration_markers: int = 6,
) -> tuple[np.ndarray | None, float | None, float | None, str, int, str]:
    """
    Return H, x, y, error, n_calibration_markers, led_marker_ids.

    A frame is accepted only when:
      - marker 10 or marker 11 is detected, because these locate the LED board
      - at least min_calibration_markers markers from IDs 0..9 are detected

    There is intentionally no path-length, jump-size, or trajectory outlier
    filtering here. Sharp corners in the real motion should be preserved; only
    frames with too few required markers are skipped and later interpolated.

    Empty error means success.
    """
    try:
        measure = build_measure()
        measure.collect(frame_bgr)

        ids = set(measure.detect.ids_flat())
        calibration_ids = sorted(i for i in ids if 0 <= i <= 9)
        led_ids = sorted(i for i in ids if i in (10, 11))
        led_ids_text = ";".join(str(i) for i in led_ids)

        if not led_ids:
            return None, None, None, "missing_led_marker_10_or_11", len(calibration_ids), led_ids_text

        if len(calibration_ids) < min_calibration_markers:
            return (
                None,
                None,
                None,
                f"too_few_calibration_markers_0_9:{len(calibration_ids)}<{min_calibration_markers}",
                len(calibration_ids),
                led_ids_text,
            )

        H = measure.compute_homography()
        if H is None:
            return None, None, None, "homography_failed", len(calibration_ids), led_ids_text

        led_center = localise_led(measure)
        x, y = float(led_center[0]), float(led_center[1])
        return H, x, y, "", len(calibration_ids), led_ids_text
    except Exception as exc:  # keep video processing alive even when a frame fails
        return None, None, None, f"{type(exc).__name__}: {exc}", 0, ""


def measure_video(
    video_path: Path,
    csv_path: Path,
    every: int = 1,
    max_frames: int | None = None,
    include_failed: bool = False,
    min_calibration_markers: int = 6,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 0.0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_ok = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        frame_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % every != 0:
                continue

            H, x, y, error, n_calibration_markers, led_marker_ids = measure_frame(
                frame,
                min_calibration_markers=min_calibration_markers,
            )
            ok = error == ""
            if ok:
                n_ok += 1

            # By default, leave undetected/failed frames out of the CSV.
            # The drawing step interpolates through these gaps from neighbouring
            # successful rows, which keeps the path continuous without polluting
            # the raw measurement table with blank coordinates.
            if ok or include_failed:
                row = {
                    "frame": frame_idx,
                    "time_s": f"{frame_idx / fps:.6f}" if fps > 0 else "",
                    "x": f"{x:.6f}" if x is not None else "",
                    "y": f"{y:.6f}" if y is not None else "",
                    "ok": "1" if ok else "0",
                    "error": error,
                    "n_calibration_markers_0_9": str(n_calibration_markers),
                    "led_marker_ids": led_marker_ids,
                }
                row.update(dict(zip(CSV_FIELDS[8:], flatten_homography(H))))
                writer.writerow(row)
                n_written += 1

            if (frame_idx + 1) % 50 == 0:
                print(f"processed {frame_idx + 1} frames, wrote {n_written} rows, {n_ok} ok")

    cap.release()
    print(f"done: wrote {n_written} rows to {csv_path} ({n_ok} ok)")


def load_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def valid_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    good = []
    for row in rows:
        if row.get("ok") != "1":
            continue
        if row.get("x", "") == "" or row.get("y", "") == "":
            continue
        if parse_homography(row) is None:
            continue
        good.append(row)
    return good


def interpolate_series_by_frame(
    path_by_frame: list[tuple[int, float, float]],
    H_by_frame: dict[int, np.ndarray],
    frame_idx: int,
) -> tuple[float, float, np.ndarray] | None:
    """
    Return interpolated x, y, H for frame_idx from neighbouring successful rows.

    Outside the measured range, this clamps to the nearest successful row. Between
    two successful rows, x, y and each homography matrix coefficient are linearly
    interpolated. This is a pragmatic smoothing/fill step for display, not a new
    physical measurement.
    """
    if not path_by_frame:
        return None

    frames = np.array([f for f, _, _ in path_by_frame], dtype=np.float64)
    xs = np.array([x for _, x, _ in path_by_frame], dtype=np.float64)
    ys = np.array([y for _, _, y in path_by_frame], dtype=np.float64)
    Hs = np.array([H_by_frame[int(f)] for f in frames], dtype=np.float64)

    x = float(np.interp(frame_idx, frames, xs))
    y = float(np.interp(frame_idx, frames, ys))

    H_flat = []
    Hs_flat = Hs.reshape(len(frames), 9)
    for col in range(9):
        H_flat.append(float(np.interp(frame_idx, frames, Hs_flat[:, col])))
    H = np.array(H_flat, dtype=np.float64).reshape(3, 3)

    return x, y, H


def world_points_to_image_xy(H_image_to_world: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """
    Convert Nx2 world points [x, y] into Nx2 image pixel points [x, y]
    using the inverse of the current frame homography.
    """
    H_inv = np.linalg.inv(H_image_to_world)
    pts = np.asarray(world_xy, dtype=np.float32).reshape(-1, 1, 2)
    image_xy = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    return image_xy


def draw_transparent_polyline(
    frame: np.ndarray,
    points_xy: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    if len(points_xy) < 2:
        return
    overlay = frame.copy()
    pts = np.round(points_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def draw_red_x(frame: np.ndarray, xy: np.ndarray, size: int = 10, thickness: int = 2) -> None:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    color = (0, 0, 255)  # BGR red
    cv2.line(frame, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str, margin: int = 16, box_text: str | None = None) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    # Use a stable reference string for the background box so it does not
    # resize as coordinates gain/lose digits. The drawn text itself is
    # fixed-width formatted by format_coordinate_label().
    size_text = box_text if box_text is not None else text
    (tw, th), baseline = cv2.getTextSize(size_text, font, scale, thickness)
    x = margin
    y = frame.shape[0] - margin
    pad = 6
    cv2.rectangle(
        frame,
        (x - pad, y - th - baseline - pad),
        (x + tw + pad, y + baseline + pad),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)



def format_coordinate_label(x: float, y: float, width: int = 10, decimals: int = 3) -> tuple[str, str]:
    """Return fixed-width coordinate text and a stable box-size template."""
    text = f"x={x:{width}.{decimals}f}, y={y:{width}.{decimals}f}"
    # Template is intentionally a little wider than normal values so the
    # background remains constant even when numbers become negative.
    template_number = "-" + ("8" * (width - decimals - 2)) + "." + ("8" * decimals)
    box_text = f"x={template_number}, y={template_number}"
    return text, box_text


def open_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """
    Open a VideoWriter with codec fallbacks.

    OpenCV is picky: the file extension and codec must match what the local
    build supports. If the requested suffix is unknown, use .mp4 instead of
    failing with a vague writer error.
    """
    supported_suffixes = {".mp4", ".m4v", ".mov", ".avi"}
    requested_path = Path(output_path)

    if requested_path.suffix.lower() not in supported_suffixes:
        fixed_name = requested_path.name + ".mp4" if requested_path.suffix == "" else requested_path.stem + ".mp4"
        fixed_path = requested_path.with_name(fixed_name)
        print(
            f"warning: output path '{requested_path}' has unsupported extension "
            f"'{requested_path.suffix}'. Writing '{fixed_path}' instead."
        )
        output_path = fixed_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".avi":
        codec_candidates = ["MJPG", "XVID"]
    else:
        codec_candidates = ["mp4v", "avc1", "H264", "XVID", "MJPG"]

    tried = []
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        tried.append(codec)
        if writer.isOpened():
            return writer, output_path
        writer.release()

    raise RuntimeError(
        f"Could not open video writer for '{output_path}'. Tried codecs: {', '.join(tried)}. "
        "Try an output filename ending in .mp4 or .avi, and make sure the output folder is writable."
    )

def annotate_video(
    video_path: Path,
    csv_path: Path,
    output_path: Path,
    show_full_path: bool = True,
    alpha: float = 0.5,
    thickness: int = 2,
) -> None:
    rows = load_csv(csv_path)
    good_rows = valid_rows(rows)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, output_path = open_video_writer(output_path, fps, (width, height))

    # Precompute measured world coordinates and homographies by frame. The CSV may
    # contain only successful detections; missing frames are filled by interpolation
    # during drawing.
    path_by_frame = []
    H_by_frame = {}
    for row in good_rows:
        f = int(row["frame"])
        path_by_frame.append((f, float(row["x"]), float(row["y"])))
        H_by_frame[f] = parse_homography(row)
    path_by_frame.sort(key=lambda item: item[0])

    if len(path_by_frame) < 2:
        raise RuntimeError("Need at least two successful detections to interpolate a continuous path.")

    frame_idx = -1
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        interpolated = interpolate_series_by_frame(path_by_frame, H_by_frame, frame_idx)
        if interpolated is not None:
            x_now, y_now, H = interpolated

            if show_full_path:
                # Interpolate one world point for every video frame up to the current
                # frame, so gaps left by failed detections are continuous.
                start_f = int(path_by_frame[0][0])
                draw_until = max(frame_idx, start_f)
                dense_world = []
                for f in range(start_f, draw_until + 1):
                    interp = interpolate_series_by_frame(path_by_frame, H_by_frame, f)
                    if interp is not None:
                        dense_world.append([interp[0], interp[1]])
                pts_world = np.array(dense_world, dtype=np.float32)
            else:
                start_f = int(path_by_frame[0][0])
                end_f = int(path_by_frame[-1][0])
                dense_world = []
                for f in range(start_f, end_f + 1):
                    interp = interpolate_series_by_frame(path_by_frame, H_by_frame, f)
                    if interp is not None:
                        dense_world.append([interp[0], interp[1]])
                pts_world = np.array(dense_world, dtype=np.float32)

            if len(pts_world) >= 2:
                pts_img = world_points_to_image_xy(H, pts_world)
                # Draw only the path polyline. No start arrow/marker is drawn.
                draw_transparent_polyline(frame, pts_img, color=(255, 255, 255), thickness=thickness, alpha=alpha)

            current_world = np.array([[x_now, y_now]], dtype=np.float32)
            current_img = world_points_to_image_xy(H, current_world)[0]
            draw_red_x(frame, current_img)
            label_text, label_box_text = format_coordinate_label(x_now, y_now)
            draw_label(frame, label_text, box_text=label_box_text)

        writer.write(frame)
        written += 1
        if written % 100 == 0:
            print(f"annotated {written} frames")

    cap.release()
    writer.release()
    print(f"done: wrote annotated video to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track LED through a video and draw its world-coordinate path.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("-out", type=Path, default=Path("video_led_results.csv"), help="CSV output path")
    parser.add_argument("-csv", type=Path, help="Existing CSV to use for --draw-only; defaults to -out")
    parser.add_argument("-annotated", type=Path, help="Annotated output video path, e.g. tracked.mp4")
    parser.add_argument("--measure-only", action="store_true", help="Only write the CSV; do not render annotated video")
    parser.add_argument("--draw-only", action="store_true", help="Only render annotated video from an existing CSV")
    parser.add_argument("--every", type=int, default=1, help="Measure every Nth frame. Default: 1")
    parser.add_argument("--max-frames", type=int, help="Optional debug limit for measuring")
    parser.add_argument(
        "--min-calibration-markers",
        type=int,
        default=6,
        help="Minimum number of detected calibration markers from IDs 0..9 required to accept a frame. Default: 6",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also write failed/undetected sampled frames to CSV. Default: omit them and interpolate for drawing.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Path overlay alpha. Default: 0.5")
    parser.add_argument("--thickness", type=int, default=2, help="Path line thickness in pixels. Default: 2")
    parser.add_argument(
        "--draw-complete-path",
        action="store_true",
        help="Draw the entire measured path on every frame instead of only the path up to the current frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.every < 1:
        raise SystemExit("--every must be >= 1")
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be between 0 and 1")
    if args.measure_only and args.draw_only:
        raise SystemExit("Use only one of --measure-only or --draw-only")

    csv_path = args.csv if args.csv is not None else args.out

    if not args.draw_only:
        measure_video(
            args.video,
            args.out,
            every=args.every,
            max_frames=args.max_frames,
            include_failed=args.include_failed,
            min_calibration_markers=args.min_calibration_markers,
        )
        csv_path = args.out

    if not args.measure_only:
        if args.annotated is None:
            raise SystemExit("Provide -annotated tracked.mp4, or use --measure-only")
        annotate_video(
            args.video,
            csv_path,
            args.annotated,
            show_full_path=not args.draw_complete_path,
            alpha=args.alpha,
            thickness=args.thickness,
        )


if __name__ == "__main__":
    main()