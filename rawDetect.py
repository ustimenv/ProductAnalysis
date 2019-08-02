import cv2
import numpy as np

from imgUtils import ImgUtils
from rawTrans import  RawDetectionTrans

class RawDetector:
    import cv2
    import numpy as np

    from imgUtils import ImgUtils
    from postCoolTrans import PostCoolTrans as T, PostCoolTrans

    def __init__(self, detectionMode=0):
        self.detectionMode = detectionMode
        self.Transformer = RawDetectionTrans()
        self.expectedW = 300        # approxiamate pixel size of the objects we are expecting
        self.expectedH = 300

        if self.detectionMode == 0:  # detect contours
            pass
        else:                       # detect blobs
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
            x1 = x + 30
            x2 = x1 + w + 30
            y1 = y
            y2 = y1 + h
            rois.append((x1, y1, x2, y2))
        return rois


    def _detectBlobs(self, frame, **kwargs):
        keypoints = self.blobber.detect(frame)
        return cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def getImgWithBoxes(self, sourceImg):
        hBefore, wBefore, _ = sourceImg.shape
        outputImg = np.copy(sourceImg)
        frame = self.prepare(sourceImg)
        hAfter, wAfter = frame.shape
        xOffset = int((wBefore - wAfter) / 2)
        yOffset = int((hBefore - hAfter) / 2)

        if self.detectionMode == 0:
            rois = self._detectContours(frame)
            for roi in rois:
                ImgUtils.drawRect(roi, outputImg, offset=(xOffset, yOffset) * 2)
        else:
            outputImg = self._detectBlobs(frame)
        return outputImg


def rawTest():
    D = RawDetector(detectionMode=0)
    for i in range(6, 56):
        img = cv2.imread('Raw/' + str(i) + '.png')
        imgX = RawDetectionTrans.transform(img)
        img0 = RawDetectionTrans.transformThresh(imgX, 0)
        img1 = RawDetectionTrans.transformThresh(imgX, 160)
        img2 = RawDetectionTrans.transformThresh(imgX, 255)

        while True:
            ImgUtils.show('Before', img, 0, 0)
            ImgUtils.show('So far', imgX, 350, 0)
            ImgUtils.show('0', img0, 700, 0)
            ImgUtils.show('1', img1, 1050, 0)
            ImgUtils.show('2', img2, 1400, 0)

            keyboard = cv2.waitKey(30)
            if keyboard == 27:
                break
            elif keyboard == ord('q'):
                return

if __name__ == "__main__":
    # rawTest()
    D = RawDetector(detectionMode=0)
    camera = cv2.VideoCapture()
    # s = 6
    # clock = 0
    camera.open('rtsp://Operator:PHZOperator@10.150.10.155/1')
    while True:
        # clock+=1
        _, feed = camera.read()
        if feed is None:
            continue
        # feed = cv2.imread('pstcool/3.png')
        # frame = D.getImgWithBoxes(feed)
        feed = D.resizeLiveFeed(feed)

        ImgUtils.show("Live", feed, 0, 0)
        # ImgUtils.show("X", frame, 350, 0)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
