import cv2
import numpy as np

from utils.imgUtils import ImgUtils
from track import Tracker
from detectionBase import BaseDetector


class Detector(BaseDetector):
    def __init__(self,):
        super(Detector, self).__init__(pathToCached='~/Samples/cached/Line30', pathToSamples='~/Samples/samples/Line30')

        trackerArgs = {'upperBound': 200, 'lowerBound': 70,
                       'rightBound': 9999, 'leftBound' : -9999,
                       'timeToDie': 5, 'timeToLive': 3,
                       }
        self.tracker = Tracker(**trackerArgs)
        self.sizeLower = 0
        self.sizeUpper = 0

    def transform(self, img):
        contrast = cv2.inRange(img, lowerb=(0, 0, 0), upperb=(220, 180, 180))
        cv2.bitwise_not(src=contrast, dst=contrast)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        cv2.morphologyEx(src=contrast, dst=contrast, op=cv2.MORPH_ERODE, kernel=kernel, iterations=1)
        cv2.medianBlur(src=contrast, dst=contrast, ksize=11)
        return contrast

    def resize(self, img):
        return cv2.resize(img, fx=0.5, fy=0.5, dsize=(0, 0))

    def detect(self, img):
        self.cacher.update(img)
        self.sampler.update(img)
        img = self.resize(img)

        contrast = self.transform(np.copy(img))
        contours, h = cv2.findContours(contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) < 1 or w < 100 or h < 40:
                continue
            elif h > 140:
                x1 = x
                x2 = x1 + w
                y1 = y
                y2 = y1 + int(h/2)
                rois.append([x1, y1, x2, y2])

                rois.append([x1, y2, x2, y2+int(h/2)])
            else:
                x1 = x
                x2 = x1 + w
                y1 = y
                y2 = y1 + h
                rois.append([x1, y1, x2, y2])
            # if y1 < 250 or x2 < 100:
            #     continue

        tracked, newRois = self.tracker.track(rois)
        self.numObjects = self.tracker.N

        for roi in rois:
            ImgUtils.drawRect(roi, img)
            detectedCentroid = ImgUtils.findRoiCentroid(roi)
            ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))
            ImgUtils.putText(coords=(roi[0] + 50, roi[1] + 50), text=str(roi[2] - roi[0]), img=img,
                             colour=(255, 255, 0), fontSize=3)

        for objectId, centroid in tracked.items():
            ImgUtils.drawCircle((centroid[0], centroid[1]), img)
            ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(0, 255, 0))

        # for roi in newRois:
        #     print(roi[3]-roi[1], roi[2]-roi[0])

        return img, contrast, []



if __name__ == "__main__":
    D = Detector()