import cv2
from utils.perception import get_perception_matrix
from lane_detector import process_frame

cap=cv2.VideoCapture('LaneVideo.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

ret, frame = cap.read()
# 640,480= frame.shape[:2]
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

M, Minv = get_perception_matrix(frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

prev_left_fit = None
prev_right_fit = None

while True:
    ret , frame=cap.read()
    if not ret:
        break

    result , prev_left_fit , prev_right_fit = process_frame(frame , M , Minv , prev_left_fit , prev_right_fit)
    out.write(result)

    cv2.imshow('lane detection', result)
    if cv2.waitKey(1)==27:
        break


cap.release()
out.release()
cv2.destroyAllWindows()