import matplotlib.pyplot as plt
import numpy as np
import cv2


def bounding_box(
      points: np.ndarray,
      im_shape: tuple[int, ...] | None = None,
      padding: int = 50,
) -> tuple[int, int, int, int]:
   """
   Compute an axis-aligned bounding box around a set of 2D points.

   Optionally clamps the bounding box to image dimensions.

   :param points: np.ndarray[np.float32 | np.float64] (N, 2) Input 2D point coordinates.
   :param im_shape: tuple[int, ...] | None Image shape used for boundary clamping.
   :param padding: int Padding added to all bounding box sides in pixels.

   :return: [tuple[int, int, int, int]] Bounding box coordinates (x_min, y_min, x_max, y_max).
   """
   x_min = max(int(np.min(points[:, 0])) - padding, 0)
   y_min = max(int(np.min(points[:, 1])) - padding, 0)
   x_max = int(np.max(points[:, 0])) + padding
   y_max = int(np.max(points[:, 1])) + padding

   if im_shape is not None:
      h, w = im_shape[:2]
      x_max, y_max = min(x_max, w), min(y_max, h)

   return x_min, y_min, x_max, y_max


def plot_detection(
      ax: plt.Axes,
      significant: np.ndarray,
      corners: np.ndarray,
      marker_id: int,
      colors: list[str] | tuple[str, ...] | None = None,
) -> None:
   """
   Plot a single detected marker with its outline, selected point, and marker ID.

   :param ax: plt.Axes Matplotlib axes used for plotting.
   :param significant: np.ndarray[np.float32 | np.float64] (2,) Selected marker point in image coordinates.
   :param corners: np.ndarray[np.float32 | np.float64] (4, 2) Marker corner coordinates.
   :param marker_id: int Detected marker ID.
   :param colors: list[str] | tuple[str, ...] | None Plot colors for outline, point, and text.

   :return: [None] Marker visualization added to the axes.
   """
   c = np.asarray(corners).reshape(4, 2)
   p = np.asarray(significant).ravel()

   closed = np.vstack([c, c[0]])

   if colors is None:
      colors = ["green", "red", "blue"]

   ax.plot(closed[:, 0], closed[:, 1], color=colors[0], linewidth=1)
   ax.scatter(p[0], p[1], color=colors[1], marker="x", s=80, linewidths=2)
   ax.text( p[0], p[1], str(marker_id), color=colors[2], fontsize=12, fontweight="bold", ha="left", va="bottom", )


def plot_detections(
      ax: plt.Axes,
      significants: list[np.ndarray] | np.ndarray,
      corners: list[np.ndarray] | np.ndarray,
      marker_ids: list[int] | np.ndarray,
      colors: list[str] | tuple[str, ...] | None = None,
) -> None:
   """
   Plot multiple detected markers with outlines, selected points, and marker IDs.

   :param ax: plt.Axes Matplotlib axes used for plotting.
   :param significants: list[np.ndarray] | np.ndarray (N, 2) Selected marker points in image coordinates.
   :param corners: list[np.ndarray] | np.ndarray (N, 4, 2) Marker corner coordinates.
   :param marker_ids: list[int] | np.ndarray[np.int32] (N,) Marker IDs corresponding to detections.
   :param colors: list[str] | tuple[str, ...] | None Plot colors for outline, point, and text.

   :return: [None] Marker visualizations added to the axes.
   """
   for significant, c, marker_id in zip(
         significants,
         corners,
         marker_ids,
   ):
      plot_detection( ax=ax,
         significant=significant,
         corners=c,
         marker_id=marker_id,
         colors=colors,
      )


def plot_image(
      ax: plt.Axes,
      image: np.ndarray,
      extent: tuple[int, int, int, int] | None = None,
      offset: tuple[float, float] | None = None,
      cmap: str = "gray",
) -> None:
   """
   Display an image or cropped image region on matplotlib axes.

   Supports grayscale and color images with optional coordinate offsets.

   :param ax: plt.Axes Matplotlib axes used for plotting.
   :param image: np.ndarray[np.uint8] (H, W) | (H, W, 3) | (H, W, 4) Input image.
   :param extent: tuple[int, int, int, int] | None Source image crop region (x_min, x_max, y_min, y_max).
   :param offset: tuple[float, float] | None Coordinate offset (dx, dy) applied to the displayed image.
   :param cmap: str Matplotlib colormap used for grayscale images.

   :return: [None] Image rendered onto the axes.
   """
   h, w = image.shape[:2]

   if extent is None:
      x_min, x_max, y_min, y_max = 0, w, 0, h
   else:
      x_min, x_max, y_min, y_max = extent

   # crop pixels in source/image coordinates
   cropped = image[y_min:y_max, x_min:x_max]

   dx, dy = offset if offset is not None else (0, 0)

   # display crop at shifted source coordinates
   plot_extent = [ x_min + dx, x_max + dx, y_max + dy, y_min + dy, ]

   kwargs = {"extent": plot_extent}

   if cropped.ndim == 2:
      kwargs["cmap"] = cmap

   ax.imshow(cropped, **kwargs)


def marker_points_from_corners(corners, marker_point_func):
   """
   Build the list of selected/significant marker points for plotting.

   :param corners: list[np.ndarray] Detected marker corners.
   :param marker_point_func: callable Function that extracts the selected marker point from one marker.

   :return: [list[np.ndarray]] Selected marker points.
   """
   return [marker_point_func(c.squeeze()) for c in corners]


def plot_marker_image(
      image: np.ndarray,
      corners: list[np.ndarray] | np.ndarray,
      marker_ids: list[int] | np.ndarray,
      marker_point_func: callable,
      crop: bool = False,
      padding: int = 50,
      colors: list[str] | tuple[str, ...] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
   """
   Visualize detected ArUco markers on an image.

   Optionally crops the displayed region around detected markers.

   :param image: np.ndarray[np.uint8] (H, W) | (H, W, 3) | (H, W, 4) Input image.
   :param corners: list[np.ndarray] | np.ndarray (N, 4, 2) Marker corner coordinates.
   :param marker_ids: list[int] | np.ndarray[np.int32] (N,) Marker IDs corresponding to detections.
   :param marker_point_func: callable Function used to extract a representative point from marker corners.
   :param crop: bool If True, crop the displayed region around detected markers.
   :param padding: int Padding in pixels applied to the cropped region.
   :param colors: list[str] | tuple[str, ...] | None Plot colors for outline, point, and text.

   :return: [tuple[plt.Figure, plt.Axes]]
       fig: matplotlib.figure.Figure Generated matplotlib figure.
       ax: matplotlib.axes.Axes Generated matplotlib axes.
   """
   significants = marker_points_from_corners(corners, marker_point_func)

   fig, ax = plt.subplots()

   if crop:
      all_points = np.vstack([c.squeeze() for c in corners])
      extent = bounding_box(
         all_points,
         im_shape=image.shape,
         padding=padding,
      )
      plot_image(ax=ax, image=image, extent=extent)
   else:
      plot_image(ax=ax, image=image)

   plot_detections(
      ax=ax,
      significants=significants,
      corners=corners,
      marker_ids=marker_ids,
      colors=colors,
   )

   format_detection_axes(
      ax=ax,
      xlabel="x [px]",
      ylabel="y [px]",
      title="Detected ArUco markers",
   )

   return fig, ax


def world_extent_from_origin(
      world_origin: np.ndarray,
      image_shape: tuple[int, ...],
      scale: float,
      padding: int,
) -> tuple[float, float, float, float]:
   """
   Compute world-coordinate plot extent from image shape and world origin offset.

   :param world_origin: np.ndarray[np.float32 | np.float64] (2,) World origin offset in pixel coordinates.
   :param image_shape: tuple[int, ...] Shape of the warped image.
   :param scale: float Pixel scale factor applied to world-coordinate units.
   :param padding: int Padding in pixels around the warped image.

   :return: [tuple[float, float, float, float]] World-coordinate extent (x_min, x_max, y_min, y_max).
   """
   h, w = image_shape[:2]

   x_min_world = (padding - world_origin[0]) / scale
   y_min_world = (padding - world_origin[1]) / scale

   x0 = x_min_world - padding / scale
   x1 = x_min_world + (w - padding) / scale
   y0 = y_min_world - padding / scale
   y1 = y_min_world + (h - padding) / scale

   return x0, x1, y0, y1


def transform_detections_to_world(
      corners: list[np.ndarray] | np.ndarray,
      marker_point_func: callable,
      H: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
   """
   Transform detected marker corners and representative points into world coordinates.

   :param corners: list[np.ndarray] | np.ndarray (N, 4, 2) Marker corner coordinates in image space.
   :param marker_point_func: callable Function used to extract a representative point from marker corners.
   :param H: np.ndarray[np.float32 | np.float64] (3, 3) Homography matrix.

   :return: [tuple[list[np.ndarray], list[np.ndarray]]]
       transformed_corners: list[np.ndarray[np.float32]] (4, 2) Marker corners in world coordinates.
       transformed_significants: list[np.ndarray[np.float32]] (2,) Representative marker points in world coordinates.
   """
   transformed_corners = []
   transformed_significants = []

   for c in corners:
      c_xy = c.reshape(-1, 1, 2).astype(np.float32)
      c_world = cv2.perspectiveTransform(c_xy, H).reshape(4, 2)
      transformed_corners.append(c_world)

      p = marker_point_func(c.squeeze()).reshape(1, 1, 2).astype(np.float32)
      p_world = cv2.perspectiveTransform(p, H).reshape(2)
      transformed_significants.append(p_world)

   return transformed_corners, transformed_significants


def crop_world_image(
      image_world: np.ndarray,
      transformed_corners: list[np.ndarray] | np.ndarray,
      extent: tuple[float, float, float, float],
      scale: float,
      padding: int,
) -> tuple[np.ndarray, list[float], tuple[float, float, float, float]]:
   """
   Crop a warped world-space image around transformed marker detections.

   Computes both image-space crop extents and corresponding world-coordinate limits.

   :param image_world: np.ndarray[np.uint8] (H, W, 3) Warped world-space image.
   :param transformed_corners: list[np.ndarray] | np.ndarray (N, 4, 2) Marker corners in world coordinates.
   :param extent: tuple[float, float, float, float] World-coordinate image extent (x_min, x_max, y_min, y_max).
   :param scale: float Pixel scale factor applied to world-coordinate units.
   :param padding: int Padding in pixels around the cropped region.

   :return: [tuple[np.ndarray, list[float], tuple[float, float, float, float]]]
       cropped: np.ndarray[np.uint8] (H_crop, W_crop, 3) Cropped world-space image.
       crop_extent: list[float] Cropped image extent in world coordinates.
       crop_limits: tuple[float, float, float, float] Cropping limits (x_min, x_max, y_min, y_max) in world coordinates.
   """
   x0, _, y0, _ = extent
   h, w = image_world.shape[:2]

   all_points = np.vstack(transformed_corners)

   x_mm_min = np.min(all_points[:, 0]) - padding / scale
   x_mm_max = np.max(all_points[:, 0]) + padding / scale
   y_mm_min = np.min(all_points[:, 1]) - padding / scale
   y_mm_max = np.max(all_points[:, 1]) + padding / scale

   px_min = int(np.floor((x_mm_min - x0) * scale))
   px_max = int(np.ceil((x_mm_max - x0) * scale))
   py_min = int(np.floor((y_mm_min - y0) * scale))
   py_max = int(np.ceil((y_mm_max - y0) * scale))

   px_min = max(px_min, 0)
   py_min = max(py_min, 0)
   px_max = min(px_max, w)
   py_max = min(py_max, h)

   cropped = image_world[py_min:py_max, px_min:px_max]
   crop_extent = [
      x0 + px_min / scale,
      x0 + px_max / scale,
      y0 + py_max / scale,
      y0 + py_min / scale,
   ]
   crop_limits = (x_mm_min, x_mm_max, y_mm_min, y_mm_max)

   return cropped, crop_extent, crop_limits


def plot_world_image(
      ax: plt.Axes,
      image_world: np.ndarray,
      extent: tuple[float, float, float, float],
      crop_data: tuple[np.ndarray, list[float], tuple[float, float, float, float]] | None = None,
) -> None:
   """
   Display a warped world-space image with optional cropped visualization limits.

   :param ax: plt.Axes Matplotlib axes used for plotting.
   :param image_world: np.ndarray[np.uint8] (H, W, 3) Warped world-space image.
   :param extent: tuple[float, float, float, float] World-coordinate image extent (x_min, x_max, y_min, y_max).
   :param crop_data: tuple[np.ndarray, list[float], tuple[float, float, float, float]] | None Optional cropped image data, crop extent, and crop limits.

   :return: [None] World-space image rendered onto the axes.
   """
   x0, x1, y0, y1 = extent

   if crop_data is None:
      ax.imshow(image_world, extent=[x0, x1, y1, y0])
      return

   cropped, crop_extent, crop_limits = crop_data
   x_mm_min, x_mm_max, y_mm_min, y_mm_max = crop_limits

   ax.imshow(cropped, extent=crop_extent)
   ax.set_xlim(x_mm_min, x_mm_max)
   ax.set_ylim(y_mm_max, y_mm_min)


def plot_marker_world_image(
      image_world: np.ndarray,
      world_origin: np.ndarray,
      H: np.ndarray,
      corners: list[np.ndarray] | np.ndarray,
      marker_ids: list[int] | np.ndarray,
      marker_point_func: callable,
      crop: bool = False,
      padding: int = 50,
      scale: float = 5.0,
      colors: list[str] | tuple[str, ...] | None = None,
      unit: str = "mm",
) -> tuple[plt.Figure, plt.Axes]:
   """
   Visualize warped marker detections in world-coordinate space.

   Optionally crops the displayed region around transformed detections.

   :param image_world: np.ndarray[np.uint8] (H, W, 3) Warped world-space image.
   :param world_origin: np.ndarray[np.float32 | np.float64] (2,) World origin offset in pixel coordinates.
   :param H: np.ndarray[np.float32 | np.float64] (3, 3) Homography matrix.
   :param corners: list[np.ndarray] | np.ndarray (N, 4, 2) Marker corner coordinates in image space.
   :param marker_ids: list[int] | np.ndarray[np.int32] (N,) Marker IDs corresponding to detections.
   :param marker_point_func: callable Function used to extract a representative point from marker corners.
   :param crop: bool If True, crop the displayed region around transformed detections.
   :param padding: int Padding in pixels around the cropped region.
   :param scale: float Pixel scale factor applied to world-coordinate units.
   :param colors: list[str] | tuple[str, ...] | None Plot colors for outline, point, and text.
   :param unit: str Unit label used for world-coordinate axes.

   :return: [tuple[plt.Figure, plt.Axes]]
       fig: matplotlib.figure.Figure Generated matplotlib figure.
       ax: matplotlib.axes.Axes Generated matplotlib axes.
   """
   extent = world_extent_from_origin(
      world_origin=world_origin,
      image_shape=image_world.shape,
      scale=scale,
      padding=padding,
   )

   transformed_corners, transformed_significants = transform_detections_to_world(
      corners=corners,
      marker_point_func=marker_point_func,
      H=H,
   )

   crop_data = None
   if crop:
      crop_data = crop_world_image(
         image_world=image_world,
         transformed_corners=transformed_corners,
         extent=extent,
         scale=scale,
         padding=padding,
      )

   fig, ax = plt.subplots()
   plot_world_image(ax=ax, image_world=image_world, extent=extent, crop_data=crop_data)

   plot_detections(
      ax=ax,
      significants=transformed_significants,
      corners=transformed_corners,
      marker_ids=marker_ids,
      colors=colors,
   )

   ax.scatter(0, 0, marker="+", s=120, linewidths=2, c="m")
   format_detection_axes(
      ax=ax,
      xlabel="world x ["+unit+"]",
      ylabel="world y ["+unit+"]",
      title="Warped image in world coordinates",
   )

   return fig, ax


def format_detection_axes(
      ax: plt.Axes,
      xlabel: str,
      ylabel: str,
      title: str,
) -> None:
   """
   Apply common formatting settings to detection visualization axes.

   Sets equal aspect ratio, axis labels, and plot title.

   :param ax: plt.Axes Matplotlib axes used for formatting.
   :param xlabel: str Label for the x-axis.
   :param ylabel: str Label for the y-axis.
   :param title: str Plot title.

   :return: [None] Axes formatting updated.
   """
   ax.set_aspect("equal")
   ax.set_xlabel(xlabel)
   ax.set_ylabel(ylabel)
   ax.set_title(title)
