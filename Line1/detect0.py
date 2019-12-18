import cv2
import numpy as np

from imgUtils import ImgUtils
from track import Tracker

# Raw detector


class Detector:
    def __init__(self):
        self.dim1Lower = 120
        self.dim1Upper = 300
        self.dim2Lower = 50
        self.dim2Upper = 200
        self.numObjects = -1

        trackerArgs = {'upperBound': 450, 'lowerBound': -9999,
                       'rightBound': 9999, 'leftBound' : 50,
                       'timeToDie': 1, 'timeToLive': 1,
                       'roiTrackingMode': True
                       }
        self.tracker = Tracker(**trackerArgs)

        self.counter = 10000
        self.guiMode = False
        self.averageColour = [0, 0, 0]
        self.averageSize = 0


    def transform(self, img):
        # contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        # contrast = cv2.bitwise_not(contrast)
        # contrast = cv2.boxFilter(src=contrast, ddepth=-1, ksize=(3, 17))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # contrast = cv2.morphologyEx(contrast, cv2.MORPH_DILATE, kernel, iterations=2)
        # contrast = cv2.threshold(src=contrast, maxval=255, thresh=200, type=cv2.THRESH_BINARY)[1]
        contrast = cv2.inRange(img, lowerb=(0, 0, 0), upperb=(150, 150, 150))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        contrast = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, kernel, iterations=1)
        return contrast

    def resize(self, img, ymin=0, ymax=270, xmin=0, xmax=500):
        return img[ymin:ymax, xmin:xmax]

    def detect(self, img):
        self.counter += 1
        hBefore, wBefore, _ = img.shape
        img = self.resize(img)

        contrast = self.transform(img)
        rois = DetectionUtils.detectContours(contrast,
                                             widthLower=self.dim1Lower, widthUpper=self.dim1Upper,
                                             heightLower=self.dim2Lower, heigthUpper=self.dim2Upper)
        tracked, _ = self.tracker.track(rois)
        self.numObjects = self.tracker.N
        if self.guiMode:
            for roi in rois:
                ImgUtils.drawRect(roi, img)
                detectedCentroid = ImgUtils.getCentroid(roi)
                ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))
            for objectId, centroid in tracked.items():
                ImgUtils.drawCircle((centroid[0], centroid[1]), img)
                ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(255, 0, 0))

        return img, contrast, []


    def detectDebug(self, img):
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        contrast = cv2.bitwise_not(contrast)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        contrast = cv2.morphologyEx(contrast, cv2.MORPH_DILATE, kernel, iterations=2)
        contrast = cv2.threshold(src=contrast, maxval=255, thresh=200, type=cv2.THRESH_BINARY)[1]
        rois, _ = DetectionUtils.houghDetect(contrast, 70, 140)
        return rois


class DetectionUtils:
    @staticmethod
    def houghDetect(img, radiusMin, radiusMax):
        """

        :param img: grayscale image with some circles
        :return:[bounding rectangles]
        """
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 150, param1=101, param2=11,
                                   minRadius=radiusMin, maxRadius=radiusMax)
        rois = []
        radii = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(img, center, 1, (0, 100, 100), 3)
                radius = i[2]
                if radiusMin < radius < radiusMax:
                    cv2.circle(img, center, radius, (255, 0, 255), 3)
                    roi = ImgUtils.circleToRectabgle(center, radius)
                    rois.append(roi)
                    radii.append(radius)
        return rois, radii

    @staticmethod
    def detectContours(frame, widthLower, widthUpper, heightLower, heigthUpper):
        """
        :param frame: grayscale image with some rectangular blobs
        :return: [(xmin, ymin, xmax, ymax), ...]
        """
        contours, h = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) < 1 or \
                w < widthLower or h < heightLower or \
                w > widthUpper or h > heigthUpper:
                continue
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h
            rois.append([x1, y1, x2, y2])
        return rois

