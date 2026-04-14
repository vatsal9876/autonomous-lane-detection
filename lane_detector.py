import cv2
from utils.perception import warp
from utils.threshold import get_binary_mask, get_lane_base
from utils.sliding_window import sliding_window
from utils.fit import fit_lane, create_lane_overlay

def process_frame(frame, M, Minv , prev_left_fit, prev_right_fit):
    frame = cv2.resize(frame, (640, 480))

    bird = warp(frame, M)
    binary = get_binary_mask(bird)

    left_base, right_base = get_lane_base(binary)

    lx, ly, rx, ry = sliding_window(binary, left_base, right_base)

    # print(f"Left lane points: {len(lx)}, Right lane points: {len(rx)}")

    if len(lx) < 3 or len(rx) < 3:
        # print("Not enough lane points detected, using previous fit.")
        return frame , prev_left_fit , prev_right_fit

    left_fit, right_fit = fit_lane(lx, ly, rx, ry)

    if len(lx) < 10 :
        left_fit = prev_left_fit
    else :
        prev_left_fit = left_fit

    if len(rx) < 10 :
        right_fit = prev_right_fit
    else :      prev_right_fit = right_fit


    overlay = create_lane_overlay(binary, left_fit, right_fit)

    # warp back
    overlay_warped = cv2.warpPerspective(overlay, Minv, (640,480))

    result = cv2.addWeighted(frame, 1, overlay_warped, 0.4, 0)

    return result , prev_left_fit , prev_right_fit