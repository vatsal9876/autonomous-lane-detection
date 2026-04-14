import cv2
import numpy as np

def fit_lane(lx, ly, rx, ry):
    left_fit = np.polyfit(ly, lx, 2)
    right_fit = np.polyfit(ry, rx, 2)

    return left_fit, right_fit

def create_lane_overlay(mask, left_fit, right_fit):
    # h, w = mask.shape

    plot_y = np.linspace(0, 479, 480)

    left_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    right_x = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, plot_y])))])

    pts = np.hstack((pts_left, pts_right)).astype(np.int32)

    overlay = np.zeros((480,640, 3), dtype=np.uint8)
    cv2.fillPoly(overlay, [pts], (0, 255, 0))

    return overlay