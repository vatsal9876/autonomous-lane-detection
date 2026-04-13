import cv2
import numpy as np

def sliding_window(mask, left_base, right_base):
    h, w = mask.shape

    window_height = 40
    margin = 50

    lx, ly = [], []
    rx, ry = [], []

    y = h

    while y > 0:
        y_low = y - window_height

        # LEFT
        left_window = mask[y_low:y, left_base-margin:left_base+margin]
        contours, _ = cv2.findContours(left_window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = False
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                left_base = left_base - margin + cx
                lx.append(left_base)
                ly.append(y)
                found=True
        if not found:
            pass

        # RIGHT
        right_window = mask[y_low:y, right_base-margin:right_base+margin]
        contours, _ = cv2.findContours(right_window, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                right_base = right_base - margin + cx
                rx.append(right_base)
                ry.append(y)

        y -= window_height

    return lx, ly, rx, ry