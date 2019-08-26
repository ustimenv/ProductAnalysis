import cv2
import numpy as np

from imgUtils import ImgUtils
from track import Tracker
from transform import Transformer


class Detector:
    def __init__(self, partioningRequired, expectedWidth, expectedHeight,
                 transformationTarget,
                 detectionMode=0, startNum=0, **kwargs):

        self.partioningRequired = partioningRequired
        self.detectionMode = detectionMode                                              # detect only contours for now
        self.transformer = Transformer(transformationTarget=transformationTarget)

        self.expectedWidth = expectedWidth; self.expectedHeight = expectedHeight         # expected product dimensions
        self.numObjects = -1                                                 # number of products we have counted so far
        self.tracker = Tracker(startNum=startNum, **kwargs)

        if self.detectionMode == 0:                 # detect contours
            pass
        else:                                       # detect blobs
            # params = cv2.SimpleBlobDetector_Params()
            self.blobber = cv2.SimpleBlobDetector_create()

    def transform(self, feed):
        return self.transformer.transform(feed)

    def resize(self, feed):
        return self.transformer.transformResize(feed)

    def detect(self, feed):
        if self.detectionMode == 0:
            self._detectContours(self.transform(feed))
        else:
            self._detectBlobs(feed, self.transform(feed))

    def _detectContours(self, frame):
        contours, h = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if self.partioningRequired:     # like postbake
                if len(approx) < 1 or w < self.expectedWidth*0.6 or h < self.expectedHeight*0.6:
                    continue
            else:                           # like raw
                if len(approx) < 1 or w < self.expectedWidth*0.6 or h < self.expectedHeight*0.6 \
                                   or w > self.expectedWidth*2 or h > self.expectedHeight*2 or w < 70:
                    continue
            x1 = x; x2 = x1 + w; y1 = y; y2 = y1 + h
            _rois = [(x1, y1, x2, y2)]
            for roi in _rois:
                rois.append(roi)
        return rois

    def _detectBlobs(self, frame, contrast,  **kwargs):
        keypoints = self.blobber.detect(contrast)
        # print(len(keypoints))
        cv2.drawKeypoints(frame, keypoints, np.array([]), (255, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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

        if self.detectionMode == 0:
            rois = self._detectContours(contrast)

            if self.partioningRequired:
                partitionedRois = self.partitionRoisX(rois)
                partitionedRois = self.partitionRoisY(partitionedRois)
            else:
                partitionedRois = rois

            tracked = self.tracker.track(partitionedRois)
            self.numObjects = self.tracker.N

            for roi in partitionedRois:
                ImgUtils.drawRect(roi, img, offset=(xOffset, yOffset) * 2)
                detectedCentroid = ImgUtils.getCentroid(roi)
                ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))

            for objectId, centroid in tracked.items():
                ImgUtils.drawCircle((centroid[0], centroid[1]), img)
                ImgUtils.putText(coords=centroid, text=str(objectId%1000), img=img, colour=(255, 0, 0))
        else:
            img = self._detectBlobs(contrast)

        return img


def slideshow():
    D = Detector(**{'expectedWidth': 140, 'expectedHeight': 60,        # detection params
                    'transformationTarget': 'raw',                      # select correct image transformations
                    'upperKillzone': 240, 'lowerKillzone': 80,              # select correct tracking parameters
                    'rightKillzone': 720, 'leftKillzone': 100,              # select correct tracking parameters
                    'timeToDie': 4, 'timeToLive': 4,
                    }, partioningRequired=False)

    for i in range(1, 772):
        img = cv2.imread('raw/3/' + str(i) + '.png')
        # contrast = D.transformer.transform(img)
        # img = D.getImgWithBoxes(img)
        while True:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img1 = frame[:, :, 1]
            img0 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img2 = cv2.inRange(img1, 80, 112)
            # img2 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)[1]



            # img0 = frame[:, :, 0]
            # img2 = frame[:, :, 2]


            # ImgUtils.show('Img', frame, 0, 0)
            ImgUtils.show('0', img0, 0, 0)
            ImgUtils.show('1', img1, 0, 300)
            ImgUtils.show('2', img2, 0, 600)

            # ImgUtils.show('Contr', contrast, 0, 350)
            keyboard = cv2.waitKey(30)
            if keyboard == 27:
                break
            elif keyboard == ord('q'):
                return



if __name__ == "__main__":
    slideshow()
