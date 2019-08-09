import cv2
import numpy as np

from imgUtils import ImgUtils
from track import Tracker
from transform import Transformer


class Detector:
    def __init__(self, expectedW, expectedH, hLow, wLow,  transformationTarget, detectionMode=0, **kwargs):
        self.detectionMode = detectionMode                # detect only contours for now
        self.transformer = Transformer(transformationTarget=transformationTarget)
        self.expectedW = expectedW                        # expected product dimensions
        self.expectedH = expectedH
        self.hLow = hLow                                    # floor below which we reject a detection
        self.wLow = wLow
        self.numObjects = 0                               # number of products we have counted so far
        self.tracker = Tracker(**kwargs)

        if self.detectionMode == 0:                 # detect contours
            pass
        else:                                       # detect blobs
            raise NotImplemented
            # self.blobber = cv2.SimpleBlobDetector()

    def transform(self, feed):
        return self.transformer.transform(feed)

    def resize(self, feed):
        return self.transformer.transformResize(feed)

    def detect(self, feed):
        if self.detectionMode == 0:
            self._detectContours(self.transform(feed))
        else:
            self._detectBlobs(self.transform(feed))

    def _detectContours(self, frame):
        contours, h = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        finalRois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) == 0 or w < self.wLow or h < self.hLow:
                continue
            x1 = x; x2 = x1 + w; y1 = y; y2 = y1 + h

            _rois = [(x1, y1, x2, y2)]
            for roi in _rois:
                rois.append(roi)
            finalRois = self.correctRois(rois)
        return finalRois

    def correctRois(self, rois):
        finalRois = []
        counter = 0
        for roi in rois:
            xmin, ymin, xmax, ymax = roi
            w = xmax - xmin
            h = ymax - ymin
            if h < self.expectedH:
                finalRois.append(roi)
            else:
                # while h / 2 > self.expectedH:
                #     counter += 1
                #     print('HMMM', roi)
                #     rois.remove(roi)
                #     rois.append((xmin, ymin, xmax, ymin+h/2))
                #     rois.append((xmin, ymin+h/2, xmax, ymax))
                #     h /= 2
                if counter > 1:
                    # print('wtf')
                    return finalRois
        return finalRois

    def _detectBlobs(self, frame, **kwargs):
        keypoints = self.blobber.detect(frame)
        return cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def getImgWithBoxes(self, img):
        hBefore, wBefore, _ = img.shape
        contrast = self.transform(img)
        hAfter, wAfter = contrast.shape
        xOffset = int((wBefore - wAfter) / 2)
        yOffset = int((hBefore - hAfter) / 2)

        if self.detectionMode == 0:
            rois = self._detectContours(contrast)
            tracked = self.tracker.track(rois)
            self.numObjects = self.tracker.N
            for roi in rois:
                ImgUtils.drawRect(roi, img, offset=(xOffset, yOffset) * 2)
                detectedCentroid = ImgUtils.getCentroid(roi)
                ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))

            for objectId, centroid in tracked.items():
                ImgUtils.drawCircle((centroid[0], centroid[1]), img)
                ImgUtils.putText(coords=centroid, text=str(objectId), img=img)
        else:
            img = self._detectBlobs(contrast)
        return img

