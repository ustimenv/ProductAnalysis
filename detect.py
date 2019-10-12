import cv2
import numpy as np

from imgUtils import ImgUtils
from track import Tracker
from transform import Transformer


class Detector:
    def __init__(self, dim1Lower, dim1Upper, dim2Lower, dim2Upper, name, initialRoi, **kwargs):
        self.dim1Lower = dim1Lower; self.dim1Upper = dim1Upper
        self.dim2Lower = dim2Lower; self.dim2Upper = dim2Upper
        self.transformer = eval('Transformer(name)')
        self.detect = eval('self.' + name)
        self.detectDebug = eval('self.'+name+'Debug')
        self.numObjects = -1
        self.tracker = Tracker(**kwargs)

        self.counter = 10000
        self.roiX1, self.roiY1, self.roiX2, self.roiY2 = initialRoi

    def transform(self, feed):
        return self.transformer.transform(feed)

    def postbake1(self, img):
        hBefore, wBefore, _ = img.shape
        img = self.transformer.resize(img)
        contrast = self.transform(img)
        rois = DetectionUtils.houghDetect(contrast, radiusMin=self.dim1Lower, radiusMax=self.dim1Upper)
        tracked = self.tracker.track(rois)
        self.numObjects = self.tracker.N

        for roi in rois:


            ImgUtils.drawRect(roi, img)
            detectedCentroid = ImgUtils.getCentroid(roi)
            ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))

        for objectId, centroid in tracked.items():
            ImgUtils.drawCircle((centroid[0], centroid[1]), img)
            ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(255, 0, 0))
        return img, contrast

    def raw1(self, img):
        self.counter += 1
        hBefore, wBefore, _ = img.shape
        img = self.transformer.resize(img, self.roiY1, self.roiY2, 250, 770)

        contrast = self.transform(img)
        rois = DetectionUtils.detectContours(contrast,
                                             widthLower=self.dim1Lower, widthUpper=self.dim1Upper,
                                             heightLower=self.dim2Lower, heigthUpper=self.dim2Upper)
        tracked = self.tracker.track(rois)
        self.numObjects = self.tracker.N
        for roi in rois:
            ImgUtils.drawRect(roi, img)
            detectedCentroid = ImgUtils.getCentroid(roi)
            ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))

        for objectId, centroid in tracked.items():
            ImgUtils.drawCircle((centroid[0], centroid[1]), img)
            ImgUtils.putText(coords=centroid, text=str(objectId % 1000), img=img, colour=(255, 0, 0))
        return img, contrast



    def postbake1Debug(self, feed):
        radiusMin = self.dim1Lower
        radiusMax = self.dim1Upper
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
                    dets.append(ImgUtils.circleToRectabgle(center, radius))
        return dets

    def raw1Debug(self, img):
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        contrast = cv2.bitwise_not(contrast)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        contrast = cv2.morphologyEx(contrast, cv2.MORPH_DILATE, kernel, iterations=2)
        contrast = cv2.threshold(src=contrast, maxval=255, thresh=200, type=cv2.THRESH_BINARY)[1]
        rois = DetectionUtils.houghDetect(contrast, 70, 140)
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
        return rois

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
    def _meanLines(lines):
        meanCoords = [0, 0, 0, 0]  # xmin, ymin, xmax, ymax

        _meanX = 0;        _meanY = 0

        # firstly find the mean x- and y-intercepts
        if lines is not None and len(lines) > 0:
            coordinates = []
            for line in lines:
                rho = line[0][0]; theta = line[0][1]
                a = np.math.cos(theta)
                b = np.math.sin(theta)
                x = a * rho;   y = b * rho
                _meanX += x;    _meanY += y
                coordinates.append([x, y])

            _meanX /= len(lines);        _meanY /= len(lines)

            meanCoordsCounter = [0, 0, 0, 0]    # num <X, num <Y, num >= X, num >= Y
            for x, y in coordinates:
                if x < _meanX:
                    meanCoords[0] += x;   meanCoordsCounter[0] += 1
                else:
                    meanCoords[2] += x; meanCoordsCounter[2] += 1
                if y < _meanY:
                    meanCoords[1] += y; meanCoordsCounter[1] += 1
                else:
                    meanCoords[3] += y; meanCoordsCounter[3] += 1

            try:
                for i in range(len(meanCoords)):
                    meanCoords[i] = int(meanCoords[i] / meanCoordsCounter[i])
            except:
                return [0, 0, 0, 0]
        return meanCoords

    @staticmethod
    def getBeltCoordinates(img):
        roi = [0 ,0, 0, 0]       # xmin, ymin, xmax, ymax
        gray = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
        height, width, _= img.shape
        edges = cv2.Canny(gray, 200, 600)

        linesHorizontal = cv2.HoughLines(image=edges, rho=1, theta=(np.pi/180), threshold=150,
                                         min_theta=45*(np.pi/180), max_theta=90*(np.pi/180))   #horizontal

        _roi = DetectionUtils._meanLines(linesHorizontal)
        roi[1], roi[3] = int(_roi[1]),  int(_roi[3])

        linesVertical = cv2.HoughLines(image=edges, rho=1, theta=(np.pi/180), threshold=150,
                                       min_theta=-45*(np.pi/180), max_theta=45*(np.pi/180))     # vertical
        _roi = DetectionUtils._meanLines(linesVertical)
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
                r = [x, ymin, x + targetWidth+ step, ymax]
                x += (step + targetWidth)
                partitionedRois.append(r)
        return partitionedRois


if __name__ == "__main__":
    pass
