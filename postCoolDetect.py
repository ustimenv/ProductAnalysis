import cv2
import numpy as np

from imgUtils import ImgUtils
from postCoolTrans import PostCoolTrans as T, PostCoolTrans
from track import Tracker


class PostCoolDetector:
    def __init__(self, detectionMode=0):
        self.detectionMode = detectionMode
        self.Transformer = PostCoolTrans()
        self.expectedW = 300                        # approxiamate pixel size of the objects we are expecting
        self.expectedH = 300

        self.T = Tracker()

        if self.detectionMode == 0:                 # detect contours
            pass
        else:                                       # detect blobs
            self.blobber = cv2.SimpleBlobDetector()

    def prepare(self, feed):
        return self.Transformer.transformWithThresh(feed)

    def resizeLiveFeed(self, feed):
        return self.Transformer.transformResize(feed)

    def detect(self, feed, **kwargs):
        if self.detectionMode == 0:
            self._detectContours(self.prepare(feed), **kwargs)
        else:
            self._detectBlobs(self.prepare(feed), **kwargs)

    def _detectContours(self, frame, hLow=40, wLow=40, maxWidth=100, maxHeight=100, **kwargs):
        contours, h = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) == 0 or w < wLow or h < hLow:
                continue
            x1 = x + 30; x2 = x1 + w + 30; y1 = y; y2 = y1 + h

            _rois = [(x1, y1, x2, y2)]
            for roi in _rois:
                rois.append(roi)
        return rois

    def _detectBlobs(self, frame, **kwargs):
        keypoints = self.blobber.detect(frame)
        return cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def getImgWithBoxes(self, sourceImg):
        hBefore, wBefore, _ = sourceImg.shape
        outputImg = self.resizeLiveFeed(sourceImg)
        frame = self.prepare(sourceImg)
        hAfter, wAfter  = frame.shape
        xOffset = int((wBefore - wAfter) / 2)
        yOffset = int((hBefore - hAfter) / 2)

        if self.detectionMode == 0:
            rois = self._detectContours(frame)
            self.T.track(rois)
            print(self.T.numObjects)
            for roi in rois:
                ImgUtils.drawRect(roi, outputImg , offset=(xOffset, yOffset)*2)
        else:
            outputImg = self._detectBlobs(frame)
        return outputImg



def pcdTest():
    D = PostCoolDetector(detectionMode=0)
    for i in range(1, 6):
        img = cv2.imread('pstcool/' + str(i) + '.png')
        frame = D.getImgWithBoxes(img)
        img = D.resizeLiveFeed(img)
        while True:
            ImgUtils.show('1', img, 1200, 0)
            ImgUtils.show('2', frame, 1600, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break



if __name__ == "__main__":
    # pcdTest()
    D = PostCoolDetector(detectionMode=0)
    camera = cv2.VideoCapture()
    camera.open('rtsp://Operator:PHZOperator@10.150.10.154/1')
    counter = 0
    s = 0
    while True:
        counter += 1
        _, feed = camera.read()
        if feed is None:
            continue

        # frame = D.getImgWithBoxes(feed)

        frame = D.getImgWithBoxes(feed)
        # feed = D.resizeLiveFeed(feed)
        contrast = D.prepare(feed)

        # if counter % 10 == 0:
        #     s+=1
        #     cv2.imwrite(str(s)+'.png', feed)
        #
        ImgUtils.show('Contrast', contrast, 1200, 0)
        # ImgUtils.show("Live", feed, 800, 0)
        ImgUtils.show("With boxes", frame, 1600, 0)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
