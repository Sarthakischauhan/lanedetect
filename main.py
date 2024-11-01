#! base/bin/python3

import cv2 as cv
import numpy as np


# Starting the video 
cap = cv.VideoCapture("./challenge_video.mp4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #ret,bw = cv.threshold(gray,127, 255,cv.THRESH_BINARY_INV);

    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
