#import the necessary libraries.
import cv2
import numpy as np
from collections import deque
import imutils

#initializing some value
blue_lower = (110,50,50) #defined lower range for blue color
blue_upper = (130,255,255)#defined upper range for blue color
white_board = np.zeros((471,636,3)) + 255  # to Create a white board
pts=deque()

cap=cv2.VideoCapture(0) #open webcam


while True:
    ret,frame=cap.read() 
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (11, 11), 0)# perform simple GaussianBlur to blur image
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blue_lower, blue_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find contours in the masked, then initialize
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)#grab the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))#get the centre of contour
        if radius > 20: #detect only when radius is greater than 20
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)# draw circle around contour
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)# store upcoming position of blue color in list
        else:
            pts.clear()
        for i in np.arange(1,len(pts)):
            cv2.line(frame,pts[i-1],pts[i],(0,0,255),5)
            cv2.line(white_board,pts[i-1],pts[i],(0,0,255),5)

    frame = cv2.flip(frame,1)#to flip the frame
    white_img = cv2.flip(white_board,1)
    cv2.imshow("frame",frame)
    cv2.imshow("white",white_img)#to show white board

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
