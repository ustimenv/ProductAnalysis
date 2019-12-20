import cv2
import numpy as np

from imgUtils import ImgUtils
from track import Tracker
from detectionBase import BaseDetector


class Detector(BaseDetector):
    def __init__(self,):
        super(Detector, self).__init__()
        self.transformer = None

        trackerArgs = {'upperBound': 200, 'lowerBound': 70,
                       'rightBound': 9999, 'leftBound' : -9999,
                       'timeToDie': 5, 'timeToLive': 3,
                       'roiTrackingMode': True
                       }
        self.tracker = Tracker(**trackerArgs)
        self.sizeLower = 0
        self.sizeUpper = 0

    def transform(self, img):
        # contrast = cv2.inRange(img, lowerb=(0, 0, 0), upperb=(220, 200, 200))
        # contrast = cv2.bitwise_not(contrast)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        # contrast = cv2.morphologyEx(contrast, cv2.MORPH_ERODE, kernel, iterations=1)

        # contrast = cv2.inRange(img, lowerb=(0, 130, 150), upperb=(200, 255, 255 ))
        contrast = cv2.inRange(img, lowerb=(0, 0, 0), upperb=(220, 180, 180))
        cv2.bitwise_not(src=contrast, dst=contrast)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        cv2.morphologyEx(src=contrast, dst=contrast, op=cv2.MORPH_ERODE, kernel=kernel, iterations=1)
        cv2.medianBlur(src=contrast, dst=contrast, ksize=11)
        return contrast

    def resize(self, img, ymin=400, ymax=1080, xmin=400, xmax=1920):
        return img[ymin:ymax, xmin:xmax]

    def detect(self, feed):
        # img = feed[550:, 350:-400, :]
        img = cv2.resize(feed, fx=0.5, fy=0.5, dsize=(0, 0))

        img = img[100+200:, 200:-200, :]

        contrast = np.copy(img)
        contrast = self.transform(contrast)
        contours, h = cv2.findContours(contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) < 1 or w < 100 or h < 40:
                continue
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h
            # if y1 < 250 or x2 < 100:
            #     continue
            rois.append([x1, y1, x2, y2])

        tracked, newRois = self.tracker.track(rois)
        self.numObjects = self.tracker.N

        for roi in rois:
            ImgUtils.drawRect(roi, img)
            detectedCentroid = ImgUtils.getCentroid(roi)
            ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))
            ImgUtils.putText(coords=(roi[0] + 50, roi[1] + 50), text=str(roi[2] - roi[0]), img=img,
                             colour=(255, 255, 0), fontSize=3)

        for objectId, centroid in tracked.items():
            ImgUtils.drawCircle((centroid[0], centroid[1]), img)
            ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(0, 255, 0))

        # for roi in newRois:
        #     print(roi[3]-roi[1], roi[2]-roi[0])

        return img, contrast, []

    def detectDebug(self, img):
        # img = img[350:, 350:-400, :]
        contrast = np.copy(img)
        contrast = self.transform(contrast)
        contours, h = cv2.findContours(contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) < 1 or w < 130 or h < 60:
                continue
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h
            if w > 250 or h > 250:
                print(w, h)
            if x2 < 250:  # or y2 < 100:
                continue
            targetHeight = 130
            numParts = h // targetHeight
            if numParts < 1:
                rois.append([x1, y1, x2, y2])
            else:
                step = (h % targetHeight)
                y = y1
                for i in range(0, numParts):
                    r = [x1, y, x2, y + targetHeight + step]
                    y += (step + targetHeight)
                    rois.append(r)
        return rois





if __name__ == "__main__":
    D = Detector()