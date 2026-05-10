# ArUco Homography Measurement

Simple Python project for detecting ArUco markers, computing homography, and transforming image coordinates into real world coordinates.

The project can:
- detect ArUco markers
- match markers to known world positions
- compute homography
- warp the image into world coordinates
- transform points between image space and world space
- visualize detections and warped images

---

# Example

## Original image with detected markers and Red point of interest

![Detected markers](fig1.png)

## Warped image in world coordinates and transformed Red point of interest

![Warped world image](fig2.png)

---

# Repository Structure

## `Detect.py`

Main ArUco detection module.

This module:
- creates ArUco markers
- creates sample marker images
- detects markers in grayscale images
- stores detected corners and IDs
- provides helper functions like:
  - `found()`
  - `ids_flat()`

Main class:
```python
Detect()
```

---

## `Measure.py`

Main measurement and homography module.

This module:
- loads images
- detects markers
- matches markers to known world coordinates
- computes homography matrix
- transforms points:
  - image → world
  - world → image
- warps image into world coordinate system
- visualizes detections

Main class:
```python
Measure()
```

Important methods:
```python
collect()
compute_homography()
warp_image_to_world()
transform_point_i2w()
transform_point_w2i()
plot_detection()
plot_world_detection()
```

---

## `helper_plots.py`

Visualization helper functions.

This module contains plotting utilities for:
- drawing detected markers
- plotting marker IDs
- plotting warped world images
- cropping detection regions
- formatting matplotlib plots

Mostly used internally by `Measure.py`.

---

## `Aruko_example.ipynb`

Example notebook showing:
- marker detection
- homography computation
- coordinate transformation
- plotting results

Good starting point for testing the project.

---

# Hardware / Frame

The setup uses:
- 20×20 aluminum tubing
- magnetic screw-on marker mounts
- printed ArUco markers

Fusion 360 model:

👉 https://a360.co/4eDsnkG

---

# Requirements

```bash
pip install opencv-python opencv-contrib-python matplotlib numpy
```

---

# Basic Usage

See full example in:

```bash
Aruko_example.ipynb
```

The notebook shows:
- marker detection
- homography computation
- coordinate transforms
- warped world image visualization

---

# Notes

- Minimum 4 known markers are required for homography.
- Marker IDs must match IDs in `WORLD_COORDS`.
- Coordinate units are user-defined (`mm`, `cm`, etc.).
- Uses OpenCV ArUco module.
