import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import cv2

from measure_LED import *



def build_measure():
    return Measure(WORLD_TL_COORDS, "tl")


def measure_led(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    measure = build_measure()
    measure.collect(image)
    measure.compute_homography()
    led_center = localise_led(measure)
    return measure, led_center


def write_output(output_path: Path, image_path: Path, x: float, y: float):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image", "x", "y"])
        writer.writerow([str(image_path), f"{x:.6f}", f"{y:.6f}"])


def save_plot(measure: Measure, led_center, plot_path: Path):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = measure.plot_world_detection(crop=True, scale=1.0, padding=50)
    ax.plot(led_center[0], led_center[1], "c+", markersize=10, mew=2, zorder=3)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure LED center from an image and print x, y coordinates."
    )
    parser.add_argument("image", type=Path, nargs="?", help="Image file to measure, e.g. sample.jpg")
    parser.add_argument("-plot", action="store_true", help="Save a detection plot. Requires -to.")
    parser.add_argument("-to", type=Path, help="PNG path for -plot, e.g. led_plot.png")
    parser.add_argument("-out", type=Path, help="CSV file to append image,x,y output line to.")
    parser.add_argument("-show_layout", action="store_true", help="Show the initial ArUco layout interactively")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.show_layout and args.image is None:
        fig, ax = plot_initial_layout()
        plt.show()
        return

    if args.image is None:
        raise SystemExit("Error: image is required unless using -show_layout")

    if args.plot and args.to is None:
        raise SystemExit("Error: -plot requires -to <plot.png>")
    if args.to is not None and not args.plot:
        raise SystemExit("Error: -to is only valid with -plot")

    measure, led_center = measure_led(args.image)
    x, y = float(led_center[0]), float(led_center[1])

    print(f"x={x:.6f}, y={y:.6f}")

    if args.out is not None:
        write_output(args.out, args.image, x, y)

    if args.plot:
        save_plot(measure, led_center, args.to)

    if args.show_layout:
        fig, ax = plot_initial_layout()
        plt.show()



if __name__ == "__main__":
    main()
