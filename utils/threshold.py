import cv2
import numpy as np

def get_binary_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(255 * sobelx / np.max(sobelx))

    _, mask = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)

    return mask

def get_lane_base(mask):
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0] // 2

    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    return left_base, right_base