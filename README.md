# ArUco Homography Measurement

Simple Python project for detecting ArUco markers, computing homography, and converting image coordinates into real-world coordinates.

## Features

- Detect ArUco markers
- Match markers to known world positions
- Compute homography
- Transform points between image and world space
- Warp images into world coordinates
- Visualize detections and transformed points

## Example

| Detection | Warped World View |
|---|---|
| ![Detected markers](fig1.png) | ![Warped world image](fig2.png) |

## Repository Structure

### `Detect.py`

ArUco detection utilities:

- generate markers
- detect markers
- store IDs and corners

Helper methods:

- [`found()`](./Detect.py#L85)
- [`ids_flat()`](./Detect.py#L92)

Main class:

```python
Detect()
```

---

### `Measure.py`

Homography and measurement tools:

- compute homography
- transform coordinates
- warp images
- visualize results

Main class:

```python
Measure()
```

Main methods:

- [`collect()`](./Measure.py#L42)
- [`compute_homography()`](./Measure.py#L78)
- [`warp_image_to_world()`](./Measure.py#L115)
- [`transform_point_i2w()`](./Measure.py#L150)
- [`transform_point_w2i()`](./Measure.py#L170)
- [`plot_detection()`](./Measure.py#L210)
- [`plot_world_detection()`](./Measure.py#L240)

---

### `Aruko_example.ipynb`

Example notebook demonstrating:

- marker detection
- homography computation
- coordinate transforms
- visualization

## Requirements

```bash
pip install opencv-python opencv-contrib-python matplotlib numpy
```

## Notes

- At least 4 known markers are required
- Marker IDs must match `WORLD_COORDS`
- Units are user-defined (`mm`, `cm`, etc.)
- Built with OpenCV ArUco module
