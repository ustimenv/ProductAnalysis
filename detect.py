import cv2
import numpy as np

from imgUtils import ImgUtils
from track import Tracker
from transform import Transformer


class Detector:
    def __init__(self, partioningRequired, expectedWidth, expectedHeight,
                 target, startNum=0, **kwargs):

        self.transformer = Transformer(transformationTarget=target)

        self.expectedWidth = expectedWidth; self.expectedHeight = expectedHeight         # expected product dimensions
        self.numObjects = -1                                                 # number of products we have counted so far
        self.tracker = Tracker(startNum=startNum, **kwargs)

        if target == 'raw':
            self.detect = self._detectContours
        elif target == 'postbake':
            self.detect = self._houghDetect
        else:
            raise NotImplemented

    def transform(self, feed):
        return self.transformer.transform(feed)

    def resize(self, feed):
        return self.transformer.transformResize(feed)

    def detect(self, feed):
        self._detectContours(self.transform(feed))

    def partitionRoisY(self, rois):
        partitionedRois = []
        for (xmin, ymin, xmax, ymax) in rois:
            h = ymax - ymin
            numParts = h // self.expectedHeight
            if numParts < 1:
                partitionedRois.append((xmin, ymin, xmax, ymax))
                continue
            step = (h % self.expectedHeight)
            y = ymin
            for i in range(0, numParts):
                r = [xmin, y, xmax, y + self.expectedHeight + step]
                y += (step + self.expectedHeight)
                partitionedRois.append(r)

        return partitionedRois

    def partitionRoisX(self, rois):
        partitionedRois = []
        for (xmin, ymin, xmax, ymax) in rois:
            w = xmax - xmin
            numParts = w // self.expectedWidth
            if numParts < 1:
                partitionedRois.append((xmin, ymin, xmax, ymax))
                continue

            step = (w % self.expectedWidth)
            x = xmin
            for i in range(0, numParts):
                r = [x, ymin, x + self.expectedWidth + step, ymax]
                x += (step + self.expectedWidth)
                partitionedRois.append(r)

        return partitionedRois

    def getImgWithBoxes(self, img):
        hBefore, wBefore, _ = img.shape
        contrast = self.transform(img)
        hAfter, wAfter = contrast.shape
        xOffset = int((wBefore - wAfter) / 2)
        yOffset = int((hBefore - hAfter) / 2)

        rois = self.detect(contrast)        # format in which the rois are stored depends on the detection mode
        tracked = self.tracker.track(rois)
        self.numObjects = self.tracker.N

        for roi in rois:
            ImgUtils.drawRect(roi, img, offset=(xOffset, yOffset) * 2)
            detectedCentroid = ImgUtils.getCentroid(roi)
            ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))

        for objectId, roi in tracked.items():
            ImgUtils.drawCircle((roi[0], roi[1]), img)
            ImgUtils.putText(coords=roi, text=str(objectId % 1000), img=img, colour=(255, 0, 0))
        return img

    def _detectContours(self, frame):
        """

        :param frame: grayscale image with some rectangular blobs
        :return: [(xmin, ymin, xmax, ymax), ...]
        """
        contours, h = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) < 1 or w < self.expectedWidth*0.7 or h < self.expectedHeight*0.7 or\
                    w > self.expectedWidth*1.4 or h > self.expectedHeight*2 or w < 70:
                continue
            x1 = x; x2 = x1 + w; y1 = y; y2 = y1 + h
            rois.append([x1, y1, x2, y2])
        return rois

    def _houghDetect(self, img):
        """

        :param img: grayscale image with some circles
        :return:[(xCenter, yCenter), radius]
        """
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 150, param1=100+1, param2=10+1,
                                   minRadius=40, maxRadius=150)
        roisRect = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(img, center, 1, (0, 100, 100), 3)
                radius = i[2]
                print(radius)
                cv2.circle(img, center, radius, (255, 0, 255), 3)
                roisRect.append(ImgUtils.circleToRectabgle(center, radius))
        return roisRect


if __name__ == "__main__":
    pass
