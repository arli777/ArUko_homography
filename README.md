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

| Detection + Red sample point | Warped World View + Transformed Red sample point|
|---|---|
| ![Detected markers](fig1.png) | ![Warped world image](fig2.png) |

## Repository Structure

### `Detect.py`

ArUco detection utilities:

- generate markers
- detect markers
- store IDs and corners
- helper methods: `found()`, `ids_flat()`

Main class:

```python
Detect()
```

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

```python
collect()
compute_homography()
warp_image_to_world()
transform_point_i2w()
transform_point_w2i()
plot_detection()
plot_world_detection()
```

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
