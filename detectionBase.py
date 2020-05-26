import time
from collections import deque
from os.path import expanduser
from random import randint

import cv2
import numpy as np

from utils.imgUtils import ImgUtils


class BaseDetector:
    def __init__(self, pathToCached, pathToSamples):
        self.tracker = None
        self.numObjects = -1
        self.cacher = FrameCacher(saveDir=pathToCached, capacity=10, minTimeTilNextSave=5)
        self.sampler = FrameSampler(saveDir=pathToSamples, framesPerSample=20, samplingPeriod=20*60, maxSamples=10000)

    def detect(self, image):
        raise NotImplementedError

    # Detection utils

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
                    roi = ImgUtils.findBoxAroundCircle(center, radius)
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

    @staticmethod
    def getBeltCoordinates(img):
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


# maintain a fixed-size cache of video frames for later analysis should something go wrong
class FrameCacher:
    def __init__(self, capacity, saveDir, minTimeTilNextSave=5):
        self.capacity = capacity
        self.cache = deque()
        self.timeOfLastSave = time.time()
        self.path = saveDir
        self.period = minTimeTilNextSave

    def update(self, frame):
        self.cache.append(frame)
        if len(self.cache) > self.capacity:
            self.cache.popleft()

    def write(self):
        currentTime = time.time()
        path = self.path + '/' + FrameCacher.formatTime(currentTime) + '|'
        if currentTime - self.timeOfLastSave >= self.period:
            for i, frame in enumerate(self.cache):
                path += str(i) + '.png'
                cv2.imwrite(path, frame)
            self.cache.clear()
            self.timeOfLastSave = currentTime

    @staticmethod
    def formatTime(t):
        t = time.asctime().split(' ')
        t[4], t[0], t[1], t[2], t[3] = t[0], t[1], t[2], t[3], t[4]
        return '|'.join(t) + '-'


class FrameSampler:
    def __init__(self, saveDir, samplingPeriod, maxSamples, framesPerSample):
        self.timeOfLastSave = time.time()
        self.path = saveDir
        self.period = samplingPeriod
        self.maxSamples = maxSamples
        self.index = 0
        self.currentSampleSize = framesPerSample
        self.framesPerSample = framesPerSample

    def update(self, frame):
        if self.index > self.maxSamples: return
        currentTime = time.time()

        # initiate sampling every hour or so
        if currentTime - self.timeOfLastSave >= self.period:
            self.currentSampleSize = 0
            self.timeOfLastSave = currentTime

        # we want to sample framesPerSample consecutive frames
        if self.currentSampleSize < self.framesPerSample:
            path = self.path + '/' + FrameCacher.formatTime(currentTime) + '|'
            path += str(self.index) + '.png'

            cv2.imwrite(expanduser(path), frame)
            self.currentSampleSize += 1; self.index += 1


    @staticmethod
    def _formatTIme(t):
        t = time.asctime().split(' ')
        t[4], t[0], t[1], t[2], t[3] = t[0], t[1], t[2], t[3], t[4]
        return '|'.join(t) + '-'

