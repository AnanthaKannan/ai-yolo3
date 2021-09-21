import cv2
import numpy as np
from sort import *

wht = 608
confThreshold = 0.5
nmsThreshold = 0.3
tracker = Sort()
memory = {}
count = 0

line = [(0, 190), (700, 190)]
classFile = 'yolo/coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = "C:\personal\yoloV3\yolo\yolov3_608.cfg"
modelWeight = "C:\personal\yoloV3\yolo\yolov3_608.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def findObjects(outputs, img):
    ht, wt, ct = img.shape
    bbox = []
    classIds = []
    confs = []
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

    bbox = []
    indexIDs = []
    global memory
    previous = memory.copy()
    memory = {}

    if len(dets) > 0:
        dets = np.asarray(dets)
        tracks = tracker.update(dets)
        if len(tracks) > 0:
            count = int(tracks[0][4])
        for track in tracks:
            print('track', track)
            bbox.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = bbox[-1]
        if len(bbox) > 0:
            i = int(0)
            for box in bbox:
                (x, y, w, h) = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                cv2.rectangle(img, (x, y), (x + w, y + h), (120, 0, 255), 2)

                text = "{}".format(indexIDs[i])
                cv2.putText(img, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                i += 1

def createBoxToCount(img):
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
