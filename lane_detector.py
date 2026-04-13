import cv2
from utils.perception import warp
from utils.threshold import get_binary_mask, get_lane_base
from utils.sliding_window import sliding_window
from utils.fit import fit_lane, create_lane_overlay

def process_frame(frame, M, Minv):
    frame = cv2.resize(frame, (640, 480))

    bird = warp(frame, M)
    binary = get_binary_mask(bird)

    left_base, right_base = get_lane_base(binary)

    lx, ly, rx, ry = sliding_window(binary, left_base, right_base)

    print(f"Left lane points: {len(lx)}, Right lane points: {len(rx)}")

    if len(lx) < 5 or len(rx) < 5:
        return frame  # fail-safe

    left_fit, right_fit = fit_lane(lx, ly, rx, ry)

    overlay = create_lane_overlay(binary, left_fit, right_fit)

    # warp back
    overlay_warped = cv2.warpPerspective(overlay, Minv, (640, 480))

    result = cv2.addWeighted(frame, 1, overlay_warped, 0.4, 0)

    return result