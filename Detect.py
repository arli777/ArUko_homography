import numpy as np
import cv2


class Detect:
   def __init__(self) -> None:
      """
      Initialize ArUco detector configuration and internal detection buffers.

      :return: [None] Detector initialized with OpenCV ArUco parameters.
      """
      self.corners = None
      self.ids = None
      self.rejected = None

      dict_name = getattr(cv2.aruco, "DICT_ARUCO_ORIGINAL")
      self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)

      self.parameters = cv2.aruco.DetectorParameters()
      self.parameters.cornerRefinementMethod = (
         cv2.aruco.CORNER_REFINE_SUBPIX
      )
      self.parameters.cornerRefinementWinSize = 5
      self.parameters.cornerRefinementMaxIterations = 50
      self.parameters.cornerRefinementMinAccuracy = 0.01

      self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

   def create_marker(
         self,
         marker_id: int,
         size: int = 200,
   ) -> np.ndarray:
      """
      Generate an ArUco marker image.

      :param marker_id: int ArUco marker ID.
      :param size: int Output marker image size in pixels.

      :return: [np.ndarray[np.uint8]] (size, size) Generated marker image.
      """

      marker = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size)
      return marker

   def create_sample_image(
         self,
         marker_range: int = 4,
         marker_size: int = 200,
         spacing: int = 50,
   ) -> np.ndarray:
      """
      Create a sample image containing multiple ArUco markers arranged in a grid.

      :param marker_range: int Marker IDs [0 ... marker_range-1] to generate and place into the canvas.
      :param marker_size: int Size of each marker in pixels.
      :param spacing: int Pixel spacing between markers and canvas borders.

      :return: [np.ndarray[np.uint8]] (H, W) Generated grayscale marker canvas.
      """

      marker_ids = list(range(marker_range))

      cols = max(int(np.sqrt(marker_range)), 1)
      rows = int(np.ceil(marker_range / cols))

      width = cols * marker_size + (cols + 1) * spacing
      height = rows * marker_size + (rows + 1) * spacing

      canvas = np.ones((height, width), dtype=np.uint8) * 255

      for i, marker_id in enumerate(marker_ids):
         marker = self.create_marker(
            marker_id,
            marker_size
         )

         row = i // cols
         col = i % cols

         x = spacing + col * (marker_size + spacing)
         y = spacing + row * (marker_size + spacing)

         canvas[
            y:y + marker_size,
            x:x + marker_size
         ] = marker

      return canvas

   def detect(
         self,
         image_gray: np.ndarray,
   ) -> tuple[list[np.ndarray], np.ndarray | None]:
      """
      Detect ArUco markers in a grayscale image.

      :param image_gray: np.ndarray[np.uint8] (H, W) Input grayscale image.

      :return: [tuple[list[np.ndarray], np.ndarray | None]]
          corners: list[np.ndarray[np.float32]] (1, 4, 2) Detected marker corners.
          ids: np.ndarray[np.int32] (N, 1) | None Detected marker IDs.
      """
      self.corners, self.ids, self.rejected = self.detector.detectMarkers(image_gray)
      return self.corners, self.ids

   def found(self) -> bool:
      """
      Check whether any ArUco markers were detected.

      :return: [bool] True if at least one marker was detected, otherwise False.
      """
      return self.ids is not None and len(self.ids) > 0

   def ids_flat(self) -> list[int]:
      """
      Return detected marker IDs as a flat Python list.

      :return: [list[int]] (N,) Flattened detected marker IDs. Returns empty list if no markers are detected.
      """
      if self.ids is None:
         return []
      return self.ids.flatten().tolist()
