
import cv2
import numpy as np

class Yolo:
    def __init__(self, confThreshold=0.5, wht=608, nmsThreashold=0.3):
        self.modelConfiguration = "D:/coding/ai-yolo3/yolov3.cgf"
        self.modelWeight = "D:/coding/ai-yolo3/yolov3.weights"
        self.confThreshold = confThreshold
        self.nmsThreashold = nmsThreashold
        classFile = 'coco.names'
        self.classNames = []
        self.wht = wht
        with open(classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        self.net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeight)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    def findObjects(self, outputs, img):
        ht, wt, ct = img.shape
        bbox = []
        classIds = []
        confs = []
        for output in outputs:
            for det in output:
                score = det[5:]
                classId = np.argmax(score)
                confidence = score[classId]
                if confidence > self.confThreshold:
                    w, h = int(det[2]*wt), int(det[3] * ht)
                    x, y = int((det[0]*wt) - w/2), int((det[1]*ht)-h/2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        
        # used to remove dublicate
        finalResponse = []
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreashold)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            className = self.classNames[classIds[i]].upper()
            confidence = int(confs[i] * 100)
            finalResponse.append([x, y, w, h, className, confidence])
        return finalResponse

    def yoloProcess(self, img):
        blob = cv2.dnn.blobFromImage(img, 1/255, (self.wht, self.wht), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layerNames = self.net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(outputNames)
        return self.findObjects(outputs, img)
        # return img

def main():
    cap = cv2.VideoCapture(0)
    yo = Yolo()
    while True:
        _, img = cap.read()
        yoloValues = yo.yoloProcess(img)
        if len(yoloValues) > 0:
            for values in yoloValues:
                x, y, w, h, className, confidence = values
                text = f'{className} {confidence}'
                cv2.putText(img, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (100, 200, 123), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()