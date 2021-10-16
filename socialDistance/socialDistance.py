import cv2
import numpy as np
from yoloModule import Yolo

cap = cv2.VideoCapture('assert/humanTopView.mp4')
socialDistance = 70


def connectedWithCenter(x, y, w, h, targetPoints):
    currentPoint = ( int((x + (x+w))/2), int((y + (y+h))/2))
    singeTargePoint = centerPoint
    distanceState = 100000
    for targetPoint in targetPoints:
        x1, y1= currentPoint
        x2, y2 = targetPoint
        distance = ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)
  
        if distanceState > distance and distance > 0:
            distanceState = distance
            singeTargePoint = targetPoint
        
    cv2.line(img, currentPoint, singeTargePoint, (200, 255, 200), 2)
    cv2.putText(img, str(int(distanceState)), currentPoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if socialDistance > int(distanceState):
         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    

yo = Yolo()
while True:
    _, img = cap.read()

    yoloValues = yo.yoloProcess(img)
    if len(yoloValues) > 0:
        centerPoints = []
        for values in yoloValues:
            x, y, w, h, className, confidence = values
            if className == 'PERSON':
                centerPoint = (int((x + (x+w))/2), int((y + (y+h))/2))
                cv2.circle(img, centerPoint, 2, (255, 255, 125), 2)
                centerPoints.append(centerPoint)

        for values in yoloValues:
            x, y, w, h, className, confidence = values
            if className == 'PERSON':
                text = f'{className} {confidence}'
                cv2.putText(img, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (100, 200, 123), 2)
                connectedWithCenter(x, y, w, h, centerPoints)
                

    cv2.imshow('Window', img)
    cv2.waitKey(1)