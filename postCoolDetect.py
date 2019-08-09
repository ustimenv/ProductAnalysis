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
        self.N = 0
        self.T = Tracker()
        if self.detectionMode == 0:                 # detect contours
            pass
        else:                                       # detect blobs
            self.blobber = cv2.SimpleBlobDetector()

    def transform(self, feed):
        return self.Transformer.transformWithThresh(feed)

    def resize(self, feed):
        return self.Transformer.transformResize(feed)

    def detect(self, feed, **kwargs):
        if self.detectionMode == 0:
            self._detectContours(self.transform(feed), **kwargs)
        else:
            self._detectBlobs(self.transform(feed), **kwargs)

    def _detectContours(self, frame, hLow=40, wLow=40, maxWidth=100, maxHeight=100, **kwargs):
        contours, h = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        finalRois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) == 0 or w < wLow or h < hLow:
                continue
            x1 = x; x2 = x1 + w; y1 = y; y2 = y1 + h

            _rois = [(x1, y1, x2, y2)]
            for roi in _rois:
                rois.append(roi)
            finalRois = self.correctRois(rois)
        return finalRois

    def correctRois(self, rois, H=200):
        finalRois = []
        counter = 0
        for roi in rois:
            xmin, ymin, xmax, ymax = roi
            w = xmax - xmin
            h = ymax - ymin
            if h < H:
                finalRois.append(roi)
            else:
                while h / 2 > H:
                    counter += 1
                    print('HMMM', roi)
                    rois.remove(roi)
                    rois.append((xmin, ymin, xmax, ymin+h/2))
                    rois.append((xmin, ymin+h/2, xmax, ymax))
                    h /= 2
                if counter > 100:
                    print('wtf')
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
            tracked = self.T.track(rois)
            # print(self.T.N)
            for roi in rois:
                ImgUtils.drawRect(roi, img, offset=(xOffset, yOffset) * 2)
                detectedCentroid = ImgUtils.getCentroid(roi)
                ImgUtils.drawCircle(detectedCentroid, img, colour=(255, 0, 0))
                # print(str(enumCentroids[objectId]))

            for objectId, centroid in tracked.items():
                ImgUtils.drawCircle((centroid[0], centroid[1]), img)
                ImgUtils.putText(coords=centroid, text=str(objectId), img=img)
        else:
            img = self._detectBlobs(contrast)
        return img


def pcdTest():
    D = PostCoolDetector(detectionMode=0)
    for i in range(1, 6):
        img = cv2.imread('pstcool/' + str(i) + '.png')
        frame = D.getImgWithBoxes(img)
        img = D.resize(img)
        while True:
            ImgUtils.show('1', img, 1200, 0)
            ImgUtils.show('2', frame, 1600, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break


def pcdTestSlideshow():
    D = PostCoolDetector(detectionMode=0)
    for i in range(1, 348):
        img = cv2.imread('pstcool/2/' + str(i) + '.png')
        # img = D.resize(img)
        frame = D.getImgWithBoxes(np.copy(img))

        while True:
            # ImgUtils.show('1', img, 1100, 0)
            ImgUtils.show('0', frame, 1500, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 27:
                break
            if keyboard == ord('q'):
                return


def pcdTestVideo():
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
        feed = D.resize(feed)
        frame = D.getImgWithBoxes(np.copy(feed))
        # contrast = D.prepare(feed)
        # if counter % 1 == 0:
        #     s+=1
        #     cv2.imwrite(str(s)+'.png', feed)

        # ImgUtils.show('Contrast', contrast, 1200, 0)
        ImgUtils.show("Live", feed, 1200, 0)
        ImgUtils.show("With boxes", frame, 1600, 0)
        print(D.T.N)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


if __name__ == "__main__":
    # pcdTestSlideshow()
    pcdTestVideo()
    # print('wowowo\n')