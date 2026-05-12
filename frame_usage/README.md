# ArUco LED Measurement

Detect ArUco markers, compute image to world homography, and localise LED position.


## Install

```bash
pip install opencv-python opencv-contrib-python matplotlib numpy
```

## Usage

| Command                                                                        | Description                    |
|--------------------------------------------------------------------------------| ------------------------------ |
| `python measure_LED_cli.py sample.jpg`                                         | Measure LED position           |
| `python measure_LED_cli.py sample.jpg -plot -to led_plot.png`                  | Save detection plot            |
| `python measure_LED_cli.py sample.jpg -out results.csv`                        | Export coordinates to CSV      |
| `python measure_LED_cli.py sample.jpg -plot -to led_plot.png -out results.csv` | Save plot and export CSV       |
| `python measure_LED_cli.py -show_layout`                                       | Show calibration marker layout |

Example output:
```text id="9jok0q"
x=534.281921, y=482.184387
```

| Flag   | `-out results.csv`                                | `-plot -to led_plot.png`             | `-show_layout`                     |
| ------ | ------------------------------------------------- |--------------------------------------|------------------------------------|
| Output | `image,x,y`<br>`sample.jpg,534.281921,482.184387` | <img src="led_plot.png" width="305"> | <img src="layout.png" width="325"> |




## 3D Models

Printable mounting models:

* [PCB ArUco holder](https://a360.co/4nsENOI) (markers **10** and **11**)

* [20×20 mm aluminium frame ArUco holder](https://a360.co/4eDsnk) (markers **0** to **9**)
  

Printed parts use: magnets `9.75×4.75×1.75` and `9×4.6×2.6`, screws `M3×12` and threaded inserts `M3 8×4.2 mm`


