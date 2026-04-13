import cv2
from utils.perception import get_perception_matrix
from lane_detector import process_frame

cap=cv2.VideoCapture('LaneVideo.mp4')

M , Minv= get_perception_matrix()

while True:
    ret , frame=cap.read()
    if not ret:
        break

    result = process_frame(frame , M , Minv)

    cv2.imshow('lane detection', result)
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()