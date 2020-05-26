import cv2
import numpy as np

from utils.imgUtils import ImgUtils
from track import Tracker
from detectionBase import BaseDetector

# Raw detector


class Detector(BaseDetector):
    def __init__(self):
        super(Detector, self).__init__()
        self.dim1Lower = 40
        self.dim1Upper = 100
        self.dim2Lower = 50
        self.dim2Upper = 200
        self.numObjects = -1

        trackerArgs = {'upperBound': 9999, 'lowerBound': -9999,
                       'rightBound': 9999, 'leftBound' : -9999,
                       'timeToDie': 0, 'timeToLive': 0,
                       }
        self.tracker = Tracker(**trackerArgs)

        self.clock = 1
        self.guiMode = False
        self.averageColour = [0, 0, 0]
        self.averageSize = 0

        self.beltXmin = 0
        self.beltXmax = 500

    def transform(self, img):
        # contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        # contrast = cv2.bitwise_not(contrast)
        # contrast = cv2.boxFilter(src=contrast, ddepth=-1, ksize=(3, 17))
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # contrast = cv2.morphologyEx(contrast, cv2.MORPH_DILATE, kernel, iterations=2)
        # contrast = cv2.threshold(src=contrast, maxval=255, thresh=200, type=cv2.THRESH_BINARY)[1]

        # contrast = cv2.inRange(img, lowerb=(0, 0, 0), upperb=(150, 150, 150))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.inRange(src=contrast, dst=contrast, lowerb=0, upperb=160)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cv2.morphologyEx(src=contrast, dst=contrast, op=cv2.MORPH_ERODE, kernel=kernel, iterations=1)

        # contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        # contrast = cv2.medianBlur(contrast, 13, 9)
        # contrast = cv2.threshold(src=contrast, maxval=255, thresh=70, type=cv2.THRESH_BINARY)[1]
        return contrast

    def resize(self, img, ymin=0, ymax=270, xmin=0, xmax=500):
        return img[150:-100, 100:-100]

    def detectOld(self, img):
        self.clock += 1
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
                detectedCentroid = ImgUtils.findRoiCentroid(roi)
                ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))
            for objectId, centroid in tracked.items():
                ImgUtils.drawCircle((centroid[0], centroid[1]), img)
                ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(255, 0, 0))

        return img, contrast, []

    def detect(self, img):
        hBefore, wBefore, _ = img.shape
        img = self.resize(img)
        contrast = self.transform(img)
        rois, radii = DetectionUtils.houghDetect(contrast, radiusMin=self.dim1Lower, radiusMax=self.dim1Upper)
        tracked, newRois = self.tracker.track(rois)
        self.numObjects = self.tracker.N

        # Re-adjust in case the belt has moved
        if self.clock < 300:
            self.clock += 1
        else:
            xmin, xmax = DetectionUtils.getBeltCoordinates(img)
            if xmin is None and xmax is None:        # if we failed to detect any belt:
                pass
            elif xmin is None:                       # assume only the right border was detected
                self.beltXmax = xmax
            else:
                if abs(self.beltXmax - xmax) < 100 and abs(self.beltXmin - xmin) < 100:  # sanity check
                    self.beltXmin = xmin
                    self.beltXmax = xmax

        if self.guiMode:
            for roi in rois:
                ImgUtils.drawRect(roi, img)
                detectedCentroid = ImgUtils.findRoiCentroid(roi)
                ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))
                ImgUtils.putText(coords=(roi[0] + 50, roi[1] + 50), text=str(roi[2]-roi[0]), img=img, colour=(255, 255, 0), fontSize=3)
            for objectId, centroid in tracked.items():
                ImgUtils.drawCircle((centroid[0], centroid[1]), img)
                ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(255, 0, 0))

        out = []
        return img, contrast, out

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
                    roi = ImgUtils.findBoxAroundCircle(center, radius)
                    rois.append(roi)
                    radii.append(radius)
        return rois, radii

    @staticmethod
    def _meanLines(lines):
        xmin, xmax = 0, 0
        xMinCount, xMaxCount = 0, 0
        meanX = 0
        xs = []
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            x = rho * np.math.cos(theta)
            meanX += x
            xs.append(x)

        meanX /= len(lines)

        for x in xs:
            if x < meanX:
                xmin += x
                xMinCount += 1
            else:
                xmax += x
                xMaxCount += 1

        xmin /= xMinCount
        xmax /= xMaxCount

        return xmin, xmax

    @staticmethod
    def getBeltCoordinates(img):
        gray = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
        height, width, _ = img.shape
        edges = cv2.Canny(gray, 200, 600)

        lines = cv2.HoughLines(image=edges, rho=1, theta=(np.pi/180), threshold=150,
                               min_theta=-45*(np.pi/180), max_theta=45*(np.pi/180))     # looking for 'vertical' lines

        ImgUtils.showLines(img=img, lines=lines)
        if lines is None or len(lines) < 1: # belt detection failed completely
            return None, None

        elif len(lines) == 1:
            return None, int(lines[0][0][0] * np.math.cos(lines[0][0][1]))

        else:
            xmin, xmax = DetectionUtils._meanLines(lines)
        return int(xmin), int(xmax)



    # @staticmethod
    # def detectContours(frame, widthLower, widthUpper, heightLower, heigthUpper):
    #     """
    #     :param frame: grayscale image with some rectangular blobs
    #     :return: [(xmin, ymin, xmax, ymax), ...]
    #     """
    #     contours, h = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     rois = []
    #     for c in contours:
    #         approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    #         x, y, w, h = cv2.boundingRect(c)
    #         if len(approx) < 1 or \
    #             w < widthLower or h < heightLower or \
    #             w > widthUpper or h > heigthUpper:
    #             continue
    #         x1 = x
    #         x2 = x1 + w
    #         y1 = y
    #         y2 = y1 + h
    #         rois.append([x1, y1, x2, y2])
    #     return rois
    #

