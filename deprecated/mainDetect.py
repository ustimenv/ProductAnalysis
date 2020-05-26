import cv2

from ohtaDetect import StandardDetector
from ohtaTrans import StandardDetectionTrans
from imgUtils import ImgUtils
from track import Tracker


class Detection:
    camera = cv2.VideoCapture()
    camera2 = cv2.VideoCapture()
    D = StandardDetector()
    T = Tracker()
    liveFeed = None
    frame = None
    rois = None

    def __init__(self):
        # self.camera.open('rtsp://operator:Operator@10.110.1.55/1')
        # self.camera.open('rtsp://Operator:PHZOperator@10.150.10.154/1')
        self.camera.open('rtsp://Operator:PHZOperator@10.150.10.155/1')

        # self.camera.open('rtsp://Kornfeil:Kornfeil7@10.150.10.153 /1')

    def main(self, draw=False):
        clock = 0
        while True:
            clock += 1
            _, self.liveFeed = self.camera.read()
            if self.liveFeed is None:
                continue

            self.frame = StandardDetectionTrans.prepareResized(self.liveFeed)
            # detect
            contrast = StandardDetectionTrans.prepareMono(self.liveFeed)

            rois = self.D.detect(contrast)
            if draw:
                for roi in rois:
                    ImgUtils.drawRect(roi, self.frame, colour=(0, 255, 255))

            # track
            self.T.track(rois)
            print('--', self.T.N)

            ###
            ImgUtils.show("Live", self.frame, 800, 0)
            ImgUtils.show("Frame", contrast, 0, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    def showFeed(self):
        while True:
            _, feed = self.camera.read()
            if feed is None:
                continue
            # self.frame = ImageTransforms.prepareResized(self.liveFeed)
            # contrast = ImageTransforms.prepareMono(self.liveFeed)

            ImgUtils.show("1", feed, 0, 0)

            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    #
    #
    # def _sample(self):
    #     yScale = self.liveFeed.shape[0] - self.frame.shape[0]
    #     xScale = self.liveFeed.shape[1] - self.frame.shape[1]
    #     for (xmin, ymin, xmax, ymax) in self.rois:
    #         ymin += int(yScale / 2)
    #         ymax += int(yScale / 2)
    #         xmin += int(xScale / 2)
    #         xmax += int(xScale / 2)
    #         self.S.extract(self.liveFeed[ymin:ymax, xmin:xmax])
    #         # ImgUtils.drawRect((xmin, ymin, xmax, ymax), liveFeed, (255, 0, 255))

    def test(self):
        while True:
            _, feed = self.camera.read()
            if feed is None:
                continue

            frame = StandardDetectionTrans.prepareMono(feed)
            for i in range(3):
                ImgUtils.show(str(i), frame[:, :, i], 0, 300*i)

            # ImgUtils.show('sum', cv2.bitwise_or(frame[:, :, 0], frame[:, :, 1]), 0, 800)
            ImgUtils.show('Feed', StandardDetectionTrans.prepareResized(feed), 800, 0)

            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break


if __name__ == "__main__":
    x = Detection()
    # x.main(draw=True)
    x.showFeed()