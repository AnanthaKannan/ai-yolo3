import cv2
import numpy as np
from sort import *

wht = 608
confThreshold = 0.5
nmsThreshold = 0.3
tracker = Sort()

classFile = 'yolo/coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


def empty(a):
    pass

"create track bar always this values statr from zero"
# windowName = "TrackBars"
# cv2.namedWindow(windowName)
# cv2.resizeWindow(windowName, 640, 240)
# cv2.createTrackbar("cx", windowName, 0, 1000, empty)
# cv2.createTrackbar("cy", windowName, 0, 1000, empty)
# cv2.createTrackbar("h", windowName, 0, 1000, empty)
# cv2.createTrackbar("w", windowName, 0, 1000, empty)

modelConfiguration = "C:\personal\yoloV3\yolo\yolov3_608.cfg"
modelWeight = "C:\personal\yoloV3\yolo\yolov3_608.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

count = 0
def findObjects(outputs, img):
    ht, wt, ct = img.shape
    bbox = []
    classIds = []
    confs = []
    global count
    cv2.line(img, (0, 160), (698, 160), (255, 0, 255), 2)
    cv2.line(img, (0, 180), (698, 180), (255, 0, 0), 2)
    # print('count', count)
    for output in outputs:
        for det in output:
            score = det[5:]
            classId = np.argmax(score)
            confidence = score[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wt), int(det[3] * ht)
                x, y = int((det[0]*wt) - w/2), int((det[1]*ht)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(confs)
    inidces = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    status = False
    dets = []
    for i in inidces:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            dets.append([x, y, x + w, y + h])
            cv2.circle(img, (x + w, y + h), 4, (0, 255, 255), 2)
            # cv2.circle(img, (350, 350), 4, (0, 255, 255), 2)

            # bbox on car
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            # text = f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}{count}%'
            # if y+h >= 160 and y+h <= 180:
            #     count += 1
                # bbox on car
            cv2.rectangle(img, (x, y), (x + w, y + h), (120, 0, 255), 2)
            if y+h <= 160:
                # count only show after cross the line
                text = f'{count}'
                cv2.putText(img, text, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # print(dets)
    if len(dets) > 0:
        dets = np.asarray(dets)
        # print(dets)
        tracks = tracker.update(dets)
        if len(tracks) > 0:
            count = int(tracks[0][4])
            print(tracks[0][4])


def createBoxToCount(img):
    # cx = cv2.getTrackbarPos("cx", windowName)
    # cy = cv2.getTrackbarPos("cy", windowName)
    # h = cv2.getTrackbarPos("h", windowName)
    # w = cv2.getTrackbarPos("w", windowName)
    cx, cy, h, w = 166, 104, 698, 350
    cv2.rectangle(img, (cx, cy), (cx+h, cy+w), (0, 255, 0), 3)
    # cv2.line(img, (cx, cy), (cx+h, cy), (0, 0, 255), 3)
    croppedImg = img[cy:cy + w, cx:cx + h]
    return croppedImg

cap = cv2.VideoCapture("carsStudyCam.mp4")
cap.set(3, 1920)
cap.set(4, 1080)
while True:
    _, img = cap.read()
    # img = cv2.imread('multicars.jpg')
    f=0.29
    img = cv2.resize(img, dsize=(0, 0), fx=f, fy=f)
    croppedImg = createBoxToCount(img)

    # net work only accept blob format (direct img format will not accept)
    blob = cv2.dnn.blobFromImage(croppedImg, 1/255, (wht, wht), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, croppedImg)
    cv2.putText(img, f'count {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# https://pjreddie.com/darknet/yolo/
# https://www.youtube.com/watch?v=1FJWXOO1SRI&list=RDCMUCYUjYU5FveRAscQ8V21w81A&index=2
