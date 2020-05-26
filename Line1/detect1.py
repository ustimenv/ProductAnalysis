import cv2
import numpy as np

from utils.imgUtils import ImgUtils
from track import Tracker
from detectionBase import BaseDetector


class Detector(BaseDetector):
    def __init__(self):
        super(Detector, self).__init__()
        # self.sizeLower = 40
        # self.sizeUpper = 150
        self.sizeLower = 60
        self.sizeUpper = 200

        trackerArgs = { 'upperBound': 300, 'lowerBound': 220,
                        'rightBound': 270, 'leftBound': 30,
                        'timeToDie': 1, 'timeToLive': 0,
         }

        self.tracker = Tracker(**trackerArgs)

        self.guiMode = False
        self.averageColour = [0, 0, 0]
        self.averageSize = 0

    def transform(self, img):
        # contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        # contrast = cv2.medianBlur(contrast, 7, 5)
        # contrast = cv2.threshold(src=contrast, maxval=255, thresh=70, type=cv2.THRESH_BINARY)[1]
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        contrast = cv2.medianBlur(contrast, 13, 9)
        contrast = cv2.threshold(src=contrast, maxval=255, thresh=70, type=cv2.THRESH_BINARY)[1]
        return contrast

    def resize(self, img, ymin=150, ymax=1000, xmin=530, xmax=830):
        return img[ymin:ymax, xmin:xmax]

    def detect(self, img):
        hBefore, wBefore, _ = img.shape
        img = self.resize(img)
        origImg = np.copy(img)
        contrast = self.transform(img)
        rois, radii = DetectionUtils.houghDetect(contrast, radiusMin=self.sizeLower, radiusMax=self.sizeUpper)
        tracked, newRois = self.tracker.track(rois)
        self.numObjects = self.tracker.N

        if self.guiMode:
            for roi in rois:
                ImgUtils.drawRect(roi, img)
                detectedCentroid = ImgUtils.findRoiCentroid(roi)
                ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))
                ImgUtils.putText(coords=(roi[0] + 50, roi[1] + 50), text=str(roi[2]-roi[0]), img=img, colour=(255, 255, 0), fontSize=3)

            for objectId, centroid in tracked.items():
                ImgUtils.drawCircle((centroid[0], centroid[1]), img)
                ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(255, 0, 0))

        # out = []

        # for roi in newRois:
            # colour = self.colour(origImg[roi[1]:roi[3], roi[0]:roi[2]])
            # self.averageColour[0] += colour[0] self.averageColour[1] += colour[1] self.averageColour[2] += colour[2]
            # self.averageSize += roi[3]-roi[1]

        return img, contrast, []


    def detectDebug(self, feed):
        radiusMin = self.sizeLower
        radiusMax = self.sizeUpper
        img = self.transform(feed)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 150, param1=101, param2=11,
                                   minRadius=radiusMin, maxRadius=radiusMax)
        dets = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(img, center, 1, (0, 100, 100), 3)
                radius = i[2]
                if radiusMin < radius < radiusMax:
                    cv2.circle(img, center, radius, (255, 0, 255), 3)
                    dets.append(ImgUtils.findBoxAroundCircle(center, radius))
        return dets


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
