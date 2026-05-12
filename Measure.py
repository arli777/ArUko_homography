import numpy as np
import cv2
import matplotlib.pyplot as plt
import helper_plots as hp

from Detect import Detect


def corners_to_point(
      corners: np.ndarray, pos: str = "tl",
) -> np.ndarray:
   """
   Extract a selected point from detected marker corners.

   Supported positions:
   - "tl": top-left
   - "tr": top-right
   - "br": bottom-right
   - "bl": bottom-left
   - "c": marker center

   :param corners: np.ndarray[np.float32] (4, 2) Marker corner coordinates.
   :param pos: str Marker reference position.

   :return: [np.ndarray[np.float32]] (2,) Selected marker point in image coordinates.
   """
   pos_map = {"tl": 0, "tr": 1, "br": 2, "bl": 3}

   if pos == "c":
      return np.mean(corners, axis=0)

   if pos in pos_map:
      return corners[pos_map[pos], :]

   raise ValueError(
      f"Unsupported position '{pos}'. "
      f"Use tl/tr/br/bl/c"
   )

class Measure:
   def __init__(
         self,
         WORLD_COORDS: None | dict[int, np.ndarray] = None,
         pos: str = "tl",
         unit: str = "mm",
   ) -> None:
      """
      Initialize ArUco-based homography measurement and world-coordinate transformation utilities.

      :param WORLD_COORDS: dict[int, np.ndarray[np.float32 | np.float64]] | None Mapping of marker IDs to world-space coordinates (2,) .
      :param pos: str Marker reference position ("tl", "tr", "br", "bl", or "c").
      :param unit: str Unit label used for world-coordinate measurements.

      :return: [None] Measurement state and calibration buffers initialized.
      """
      self.world = WORLD_COORDS  # expects dict{id: np.ndarray. shape (2, )} in mm
      self.pos = pos.lower()
      self.unit = unit

      self.image_bgr = None
      self.image_gray = None

      self.detect = Detect()

      self.calibration_indices = []
      self.unused_indices = []
      self.image_points = []
      self.world_points = []

      self.H = np.array([])
      self.homography_mask = None

      self.image_bgr_world = None
      self.world_origin = None

   def format_image(self, image: np.ndarray) -> None:
      """
      Convert an input image to internal grayscale and BGR representations.

      Supports grayscale, RGB, and RGBA image formats.

      :param image: np.ndarray[np.uint8] (H, W) | (H, W, 3) | (H, W, 4) Input image.

      :return: [None] Internal self.image_gray and self.image_bgr images are updated.
      """
      if image.ndim == 2:
         self.image_gray = image
         self.image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

      elif image.shape[2] == 4:
         self.image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
         self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

      elif image.shape[2] == 3:
         self.image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
         self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

      else:
         raise ValueError("Unsupported image format")

   def collect(
         self,
         image: np.ndarray,
   ) -> tuple[list[np.ndarray], np.ndarray | None]:
      """
      Format an input image and detect ArUco markers.

      :param image: np.ndarray[np.uint8] (H, W) | (H, W, 3) | (H, W, 4) Input grayscale, RGB, or RGBA image.

      :return: [tuple[list[np.ndarray], np.ndarray | None]]
          corners: list[np.ndarray[np.float32]] (1, 4, 2) Detected marker corners.
          ids: np.ndarray[np.int32] (N, 1) | None Detected marker IDs.
      """
      self.format_image(image)
      return self.detect.detect(self.image_gray)


   def marker_point(
         self,
         corners: np.ndarray,
   ) -> np.ndarray:
      """
      Extract a selected point from detected marker corners.

      Supported positions:
      - "tl": top-left
      - "tr": top-right
      - "br": bottom-right
      - "bl": bottom-left
      - "c": marker center

      :param corners: np.ndarray[np.float32] (4, 2) Marker corner coordinates.

      :return: [np.ndarray[np.float32]] (2,) Selected marker point in image coordinates.
      """
      return corners_to_point(corners, self.pos)

   def find_calibration_matches(
         self,
   ) -> tuple[np.ndarray, np.ndarray]:
      """
      Match detected marker IDs with known world coordinates and collect calibration point pairs.

      Stores matched indices, ignored marker indices, image points, and world points
      for homography calibration.

      :return: [tuple[np.ndarray, np.ndarray]]
          image_points: np.ndarray[np.float32] (N, 2) Calibration points in image coordinates.
          world_points: np.ndarray[np.float32] (N, 2) Corresponding calibration points in world coordinates.
      """

      self.calibration_indices = []
      self.unused_indices = []
      self.image_points = []
      self.world_points = []

      if not self.detect.found():
         print("No markers detected")
         return [], []

      ids = self.detect.ids_flat()

      for i, marker_id in enumerate(ids):
         if marker_id not in self.world:
            self.unused_indices.append(i)
            continue

         corners = self.detect.corners[i].squeeze()  # shape (4, 2)
         cal_point = self.marker_point(corners)

         self.image_points.append(cal_point)
         self.world_points.append(self.world[marker_id][::-1])
         self.calibration_indices.append(i)

      self.image_points = np.array(self.image_points, dtype=np.float32)
      self.world_points = np.array(self.world_points, dtype=np.float32)

      return self.image_points, self.world_points

   def compute_homography_from_matches(
         self,
         method: int = cv2.RANSAC,
   ) -> np.ndarray | None:
      """
      Compute homography matrix from matched image and world calibration points.

      Requires at least 4 matched calibration points.

      :param method: int OpenCV homography estimation method (e.g. cv2.RANSAC).

      :return: [np.ndarray[np.float64]] (3, 3) Homography matrix, or None if computation fails.
      """

      self.H = None
      self.homography_mask = None

      if len(self.calibration_indices) < 4:
         print("Need at least 4 matched markers for homography")
         return None

      self.H, self.homography_mask = cv2.findHomography(
         self.image_points,
         self.world_points,
         method=method
      )

      if self.H is None:
         print("Homography computation failed")
         return None

      return self.H

   def warp_image_to_world(
         self,
         scale: float = 50.,
         padding: int = 50,
   ) -> np.ndarray:
      """
      Warp the loaded image into world-coordinate space using the computed homography.

      Stores the warped image and world origin offset internally.

      :param scale: float Pixel scale factor applied to world-coordinate units.
      :param padding: int Padding in output pixels around the warped image.

      :return: [np.ndarray[np.uint8]] (H_world, W_world, 3) Warped BGR image in world-coordinate space.
      """

      if self.image_bgr is None:
         raise ValueError("No image loaded. Run collect(image) first.")

      if self.H is None:
         raise ValueError("Homography not computed. Run compute_homography() first.")

      h, w = self.image_bgr.shape[:2]

      image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32, ).reshape(-1, 1, 2)
      world_corners = cv2.perspectiveTransform(image_corners, self.H).reshape(-1, 2)

      x_min = np.min(world_corners[:, 0])
      y_min = np.min(world_corners[:, 1])
      x_max = np.max(world_corners[:, 0])
      y_max = np.max(world_corners[:, 1])

      out_w = int(np.ceil((x_max - x_min) * scale)) + 2 * padding
      out_h = int(np.ceil((y_max - y_min) * scale)) + 2 * padding

      T = np.array([[scale, 0, -x_min * scale + padding], [0, scale, -y_min * scale + padding], [0, 0, 1], ],
                   dtype=np.float64, )
      H_world_image = T @ self.H

      self.image_bgr_world = cv2.warpPerspective(self.image_bgr, H_world_image, (out_w, out_h), )
      self.world_origin = np.array([-x_min * scale + padding, -y_min * scale + padding, ], dtype=np.float32, )

      return self.image_bgr_world

   def compute_homography(self) -> np.ndarray | None:
      """
      Find calibration matches and compute homography matrix.

      :return: [np.ndarray[np.float64]] (3, 3) Homography matrix, or None if computation fails.
      """
      self.find_calibration_matches()
      return self.compute_homography_from_matches()

   def transform_point_i2w(
         self,
         points_image_yx: np.ndarray,
   ) -> np.ndarray:
      """
      Transform image-space points into world coordinates using the computed homography.
      Input and output points use (y, x) coordinate ordering.

      :param points_image_yx: np.ndarray[np.float32 | np.float64] (2, N) Image-space points in (y, x) format.

      :return: [np.ndarray[np.float32]] (2, N) Transformed world-space points in (y, x) format.
      """

      if self.H is None:
         raise ValueError("Homography not computed.  Run compute_homography() first.")

      points_image_yx = np.asarray(points_image_yx, dtype=np.float32, )

      if points_image_yx.ndim > 2 or points_image_yx.shape[0] != 2:
         raise ValueError("Expected shape (2, N)")

      points_xy = points_image_yx[::-1].T  # (y, x) -> (x, y)

      points_cv = points_xy.reshape(-1, 1, 2)  # OpenCV expects (N, 1, 2)

      world_xy = cv2.perspectiveTransform(points_cv, self.H, ).reshape(-1, 2)

      return world_xy[:, ::-1].T  # (2, N) in (y, x)

   def transform_point_w2i(
         self,
         points_world_yx: np.ndarray,
   ) -> np.ndarray:
      """
      Transform world-space points into image coordinates using the inverse homography.
      Input and output points use (y, x) coordinate ordering.

      :param points_world_yx: np.ndarray[np.float32 | np.float64] (2, N) World-space points in (y, x) format.

      :return: [np.ndarray[np.float32]] (2, N) Transformed image-space points in (y, x) format.
      """

      if self.H is None:
         raise ValueError("Homography not computed. " "Run compute_homography() first.")

      points_world_yx = np.asarray(points_world_yx, dtype=np.float32, )

      if points_world_yx.ndim > 2 or points_world_yx.shape[0] != 2:
         raise ValueError("Expected shape (2, N)")

      H_inv = np.linalg.inv(self.H)

      points_xy = points_world_yx[::-1].T  # (y, x) -> (x, y)

      points_cv = points_xy.reshape(-1, 1, 2)

      image_xy = cv2.perspectiveTransform(points_cv, H_inv, ).reshape(-1, 2)

      return image_xy[:, ::-1].T  # (2, N) in (y, x)

   def plot_detection(
         self,
         crop: bool = False,
         padding: int = 50,
         colors: list[str] | tuple[str, ...] | None = None,
   ) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
      """
      Visualize detected ArUco markers on the loaded image.

      Optionally crops the displayed region around detected markers.

      :param crop: bool If True, crop the displayed region around detected markers.
      :param padding: int Padding in pixels applied to the cropped region.
      :param colors: list[str] | tuple[str, ...] | None Plot colors for outline, point, and text.

      :return: [tuple[plt.Figure, plt.Axes] | tuple[None, None]]
          fig: matplotlib.figure.Figure Generated matplotlib figure.
          ax: matplotlib.axes.Axes Generated matplotlib axes.
      """
      if self.image_bgr is None:
         raise ValueError("No image loaded. Run collect(image) first.")

      if not self.detect.found():
         print("No markers detected")
         return None, None

      return hp.plot_marker_image(
         image=self.image_bgr,
         corners=self.detect.corners,
         marker_ids=self.detect.ids_flat(),
         marker_point_func=self.marker_point,
         crop=crop,
         padding=padding,
         colors=colors,
      )

   def plot_world_detection(
         self,
         crop: bool = False,
         padding: int = 50,
         scale: float = 5.0,
         colors: list[str] | tuple[str, ...] | None = None,
   ) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
      """
      Visualize detected ArUco markers in warped world-coordinate space.

      Generates a world-space warped image using the computed homography and
      optionally crops the displayed region around transformed detections.

      :param crop: bool If True, crop the displayed region around transformed detections.
      :param padding: int Padding in pixels around the cropped region.
      :param scale: float Pixel scale factor applied to world-coordinate units.
      :param colors: list[str] | tuple[str, ...] | None Plot colors for outline, point, and text.

      :return: [tuple[plt.Figure, plt.Axes] | tuple[None, None]]
          fig: matplotlib.figure.Figure Generated matplotlib figure.
          ax: matplotlib.axes.Axes Generated matplotlib axes.
      """

      if not self.detect.found():
         print("No markers detected")
         return None, None

      self.warp_image_to_world(scale=scale, padding=padding)

      return hp.plot_marker_world_image(
         image_world=self.image_bgr_world,
         world_origin=self.world_origin,
         H=self.H,
         corners=self.detect.corners,
         marker_ids=self.detect.ids_flat(),
         marker_point_func=self.marker_point,
         crop=crop,
         padding=padding,
         scale=scale,
         colors=colors,
         unit=self.unit,
      )


