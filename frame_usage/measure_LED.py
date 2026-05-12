import numpy as np

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import cv2

from Measure import Measure, corners_to_point

PCB_height = 24.38
PCB_width = 51.0
LED_dist_R = 30.63
LED_dist_L = 28.42
ArUco_TB_dist = 28.17
Aruco_R_dist = 13.97 - 9.2
Aruco_border_th = 2.5
Aruco_side_width = 40.

TOP_ID = 10
BOTTOM_ID = 11

N_MARKERS = 10
WORLD_TL_BORDER_COORDS =  {
    0: np.array([-9.2, 105.0]),
    1: np.array([-9.2, 1004.27]),
    2: np.array([-9.2, 1169.27]),
    3: np.array([194.93, -9.2]),
    4: np.array([842.42, -9.2]),
    5: np.array([195.57, 1260.07]),
    6: np.array([844.43, 1260.07]),
    7: np.array([1060.13, 122.35]),
    8: np.array([1060.13, 287.35]),
    9: np.array([1060.13, 1151.88]),
}
WORLD_TL_POS= {
    0: "tl",
    1: "tl",
    2: "tl",
    3: "bl",
    4: "bl",
    5: "bl",
    6: "bl",
    7: "tl",
    8: "tl",
    9: "tl",
}

ORIGIN = np.array([44.93, 44.61])
ARUCO_OFFSET = {
    "tl": np.array([1.0, 1.0]) * Aruco_border_th,
    "bl": np.array([-1.0, 1.0]) * Aruco_border_th,
}

WORLD_TL_COORDS = { i: WORLD_TL_BORDER_COORDS[i] - ORIGIN + ARUCO_OFFSET[WORLD_TL_POS[i]] for i in range(N_MARKERS)}

def plot_initial_layout( ):
   fig, ax = plt.subplots()

   ax.plot(ORIGIN[1], ORIGIN[0], "b+", markersize=10, mew=2, zorder=3)

   for i in WORLD_TL_COORDS .keys():
      y, x = WORLD_TL_COORDS [i]+ORIGIN # coords are [y, x]
      ax.plot(x, y, "rx", markersize=10, mew=2, zorder=3)
      ax.text(x - Aruco_side_width, y - Aruco_side_width, str(i), color="red", fontsize=12, zorder=4)

      by, bx = WORLD_TL_BORDER_COORDS[i]
      corner = WORLD_TL_POS[i]

      # Build square from the named corner.
      if corner == "tl":
         left = bx
         top = by
      elif corner == "bl":
         left = bx
         top = by - Aruco_side_width
      elif corner == "tr":
         left = bx - Aruco_side_width
         top = by
      elif corner == "br":
         left = bx - Aruco_side_width
         top = by - Aruco_side_width
      else:
         raise ValueError(f"Unknown corner position: {corner}")

      rect = plt.Rectangle(
         (left, top),  # plotted as (y, x)
         Aruco_side_width,
         Aruco_side_width,
         facecolor="yellow",
         edgecolor="black",
         zorder=0,
      )
      ax.add_patch(rect)

   ax.set_xlabel("x")
   ax.set_ylabel("y")
   ax.set_aspect("equal", adjustable="box")
   ax.invert_yaxis()

   ax.grid(True)
   fig.tight_layout()
   return fig, ax


def get_marker_corners_world_xy(measure: Measure, marker_id: int):
   ids = measure.detect.ids_flat()

   if marker_id not in ids:
      return None

   idx = ids.index(marker_id)

   corners_img_xy = measure.detect.corners[idx].reshape(-1, 1, 2).astype(np.float32)
   corners_world_xy = cv2.perspectiveTransform(corners_img_xy, measure.H).reshape(4, 2)

   return corners_world_xy


def params_from_corners(corners_world_xy, id: int):
   tl, tr, br, bl = corners_world_xy
   if id != TOP_ID and id != BOTTOM_ID:
      raise ValueError(f"Invalid ArUco ID: {id}")
   coef = 1. if id == TOP_ID else -1.

   l = coef*(bl - tl)
   l /= np.linalg.norm(l)

   r = coef*(br - tr)
   r /= np.linalg.norm(r)

   t = (tl - tr)
   t /= np.linalg.norm(t)

   b = (bl - br)
   b /= np.linalg.norm(b)

   y_vec = l + r
   x_vec = t + b

   y_vec = y_vec / np.linalg.norm(y_vec)
   x_vec = x_vec / np.linalg.norm(x_vec)

   start = br if id == TOP_ID else tr
   return x_vec, y_vec, start


def localise_led(measure: Measure):
   dist_y = ArUco_TB_dist / 2. + Aruco_border_th
   dist_x = (PCB_width + LED_dist_R - LED_dist_L) / 2. - Aruco_R_dist - Aruco_border_th

   top_corners_world = get_marker_corners_world_xy(measure, TOP_ID)
   bottom_corners_world = get_marker_corners_world_xy(measure, BOTTOM_ID)

   if top_corners_world is None and bottom_corners_world is None:
      raise ValueError("No LED ArUco marker found.")

   center = None

   if top_corners_world is not None:
      y_vec, x_vec, start = params_from_corners(top_corners_world, TOP_ID)
      center = start + dist_y * y_vec + dist_x * x_vec

   if bottom_corners_world is not None:
      y_vec, x_vec, start = params_from_corners(bottom_corners_world, BOTTOM_ID)
      led_b = start + dist_y * y_vec + dist_x * x_vec

      if center is not None:
         center = (center + led_b) / 2.
      else:
         center = led_b

   return center

def main():

   fig, ax = plot_initial_layout( )
   plt.show()

   measure = Measure(WORLD_TL_COORDS , "tl")

   image = cv2.imread("sample.jpg")
   measure.collect(image)
   measure.compute_homography()


   led_center = localise_led(measure)

   fig, ax = measure.plot_world_detection(crop=True, scale=1., padding=50)  # scale 50 px / mm
   ax.plot(led_center[0], led_center[1], "c+", markersize=10, mew=2, zorder=3)

   plt.show()

if __name__ == "__main__":
   main()
