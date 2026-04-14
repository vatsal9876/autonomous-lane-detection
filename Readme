# Autonomous Lane Detection (Classical Computer Vision)

## Overview

This project implements a real-time lane detection system using classical computer vision techniques. The pipeline processes video frames to detect lane boundaries and overlay the drivable region on the road.

Unlike deep learning approaches, this system relies entirely on handcrafted image processing and geometric transformations.

---

## Features

* Perspective transform (bird’s-eye view)
* Lane extraction using gradient + color thresholding
* Histogram-based lane initialization
* Sliding window tracking
* Polynomial curve fitting
* Lane area visualization
* Frame-by-frame video processing

---

## Pipeline

```
Input Frame
   ↓
Resize (640x480)
   ↓
Perspective Transform (Bird View)
   ↓
Binary Mask (Sobel + Color Threshold)
   ↓
Histogram → Lane Base Points
   ↓
Sliding Window → Lane Points
   ↓
Polynomial Fit
   ↓
Lane Region Generation
   ↓
Inverse Perspective Transform
   ↓
Overlay on Original Frame
```

---

## Project Structure

```
project/
│
├── main.py                  # Runs video pipeline
├── lane_detector.py         # Core lane detection logic
│
├── utils/
│   ├── perspective.py       # Perspective transform
│   ├── threshold.py         # Binary mask generation
│   ├── sliding_window.py    # Lane tracking
│   ├── fit.py               # Polynomial fitting
│
├── videos/
│   └── LaneVideo.mp4
│
└── output.mp4               # Generated result
```

---

## Methodology

### 1. Perspective Transform

A trapezoidal region of interest (ROI) is manually selected and warped into a rectangular bird’s-eye view. This simplifies lane geometry and makes detection easier.

### 2. Binary Masking

Lane pixels are extracted using:

* Sobel gradient (detects edges)
* Color thresholding (detects white lanes)

### 3. Histogram Initialization

A vertical histogram of the lower half of the image is used to locate the base positions of left and right lanes.

### 4. Sliding Window Search

The algorithm scans upward using windows, tracking lane pixels based on local contours.

### 5. Polynomial Fitting

Second-degree polynomials are fitted to lane points:

```
x = ay² + by + c
```

This provides smooth lane curves.

### 6. Lane Visualization

The region between left and right curves is filled and projected back onto the original frame.

---

## Challenges & Fixes

### Dashed Lane Handling

* Problem: Left lane often broken → fewer detected points
* Fixes:

  * Increased window height
  * Morphological closing
  * Temporal fallback using previous frames

### Unstable Polynomial Fit

* Cause: insufficient or noisy points
* Fix: minimum point threshold + fallback to previous fit

### Perspective Sensitivity

* Hardcoded ROI tuned for specific camera setup
* Requires consistent resolution (640x480)

---

## How to Run

```bash
python main.py
```

Press `ESC` to exit.

---

## Output

* Real-time lane overlay using OpenCV window
* Optional saved output video (`output.mp4`)

---

## Limitations

* Works best for fixed camera setups
* Sensitive to lighting and shadows
* Struggles with heavily occluded or faded lanes
* Not robust for sharp curves or extreme perspectives

---

## Future Improvements

* Lane curvature estimation
* Vehicle offset from center
* Temporal smoothing across frames
* Replace sliding window with search-around-poly
* Deep learning-based lane segmentation (U-Net / YOLOP)

---

## Key Takeaway

This project demonstrates how classical computer vision techniques can be used to detect lanes without machine learning. It highlights the transition from handcrafted feature extraction to learned representations in modern CV systems.

---

## Author

Vatsal Gupta
