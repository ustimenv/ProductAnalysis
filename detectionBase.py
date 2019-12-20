import re
import time
from datetime import datetime

import cv2
import numpy as np
from imgUtils import ImgUtils


class BaseDetector:
    def __init__(self):
        self.tracker = None
        self.numObjects = -1

        self.imgCache = dict()
        self.cacheIndex = 0

        self.historic = []

    def evaluatePast(self, image, rois):
        self.imgCache[self.cacheIndex % 20] = \
            (re.sub('\.[0-9]*', '', str(datetime.fromtimestamp(time.time()))),
             image)
        self.cacheIndex += 1

        pass

    def flushCache(self):
        # if something weird has occured, flush the cache for future evaluation
        for (tsp, img) in self.imgCache.values():
            cv2.imwrite('Imgs/' + tsp, img)
        self.imgCache.clear()

    def detect(self, image):
        raise NotImplementedError

    def detectDebug(self, image):
        raise NotImplementedError

    
    def detectContours(self, frame, widthLower, widthUpper, heightLower, heigthUpper):
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

    def houghDetect(self, img, radiusMin, radiusMax):
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
    def _meanLines(lines):
        meanCoords = [0, 0, 0, 0]  # xmin, ymin, xmax, ymax

        _meanX = 0
        _meanY = 0

        # firstly find the mean x- and y-intercepts
        if lines is not None and len(lines) > 0:
            coordinates = []
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                a = np.math.cos(theta)
                b = np.math.sin(theta)
                x = a * rho
                y = b * rho
                _meanX += x
                _meanY += y
                coordinates.append([x, y])

            _meanX /= len(lines)
            _meanY /= len(lines)

            meanCoordsCounter = [0, 0, 0, 0]  # num <X, num <Y, num >= X, num >= Y
            for x, y in coordinates:
                if x < _meanX:
                    meanCoords[0] += x
                    meanCoordsCounter[0] += 1
                else:
                    meanCoords[2] += x
                    meanCoordsCounter[2] += 1
                if y < _meanY:
                    meanCoords[1] += y
                    meanCoordsCounter[1] += 1
                else:
                    meanCoords[3] += y
                    meanCoordsCounter[3] += 1

            try:
                for i in range(len(meanCoords)):
                    meanCoords[i] = int(meanCoords[i] / meanCoordsCounter[i])
            except:
                return [0, 0, 0, 0]
        return meanCoords

    def getBeltCoordinates(self, img):
        roi = [0, 0, 0, 0]  # xmin, ymin, xmax, ymax
        gray = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
        height, width, _ = img.shape
        edges = cv2.Canny(gray, 200, 600)

        linesHorizontal = cv2.HoughLines(image=edges, rho=1, theta=(np.pi / 180), threshold=150,
                                         min_theta=45 * (np.pi / 180), max_theta=90 * (np.pi / 180))  # horizontal

        _roi = BaseDetector._meanLines(linesHorizontal)
        roi[1], roi[3] = int(_roi[1]), int(_roi[3])

        linesVertical = cv2.HoughLines(image=edges, rho=1, theta=(np.pi / 180), threshold=150,
                                       min_theta=-45 * (np.pi / 180), max_theta=45 * (np.pi / 180))  # vertical
        _roi = BaseDetector._meanLines(linesVertical)
        roi[0], roi[2] = int(_roi[0]), int(_roi[2])
        return roi

    @staticmethod
    def partitionRoisY(rois, targetHeight):
        partitionedRois = []
        for (xmin, ymin, xmax, ymax) in rois:
            h = ymax - ymin
            numParts = h // targetHeight
            if numParts < 1:
                partitionedRois.append((xmin, ymin, xmax, ymax))
                continue
            step = (h % targetHeight)
            y = ymin
            for i in range(0, numParts):
                r = [xmin, y, xmax, y + targetHeight + step]
                y += (step + targetHeight)
                partitionedRois.append(r)
        return partitionedRois

    @staticmethod
    def partitionRoisX(rois, targetWidth):
        partitionedRois = []
        for (xmin, ymin, xmax, ymax) in rois:
            w = xmax - xmin
            numParts = w // targetWidth
            if numParts < 1:
                partitionedRois.append((xmin, ymin, xmax, ymax))
                continue

            step = (w % targetWidth)
            x = xmin
            for i in range(0, numParts):
                r = [x, ymin, x + targetWidth + step, ymax]
                x += (step + targetWidth)
                partitionedRois.append(r)
        return partitionedRois


# encapsulates detection information on an image
class Detection:
    def __init__(self, rois):
        self.areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in rois]
        self.centroids = [(int((roi[0] + roi[2]) / 2.0),
                           int((roi[0] + roi[2]) / 2.0)) for roi in rois]

