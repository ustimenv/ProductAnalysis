import glob

import cv2
import matplotlib.pyplot as plt

from imgUtils import ImgUtils
from predict import *


class Manager:
    camera = cv2.VideoCapture()
    camera2 = cv2.VideoCapture()
    P = Predictor()
    rawFrame = None
    frame = None
    rois = None

    def __init__(self):
        pass
        # self.camera.open('rtsp://operator:Operator@10.110.1.55/1')
        # self.camera.open('rtsp://Operator:PHZOperator@10.150.10.154/1')
        # self.camera2.open('rtsp://Operator:PHZOperator@10.150.10.155/1')
        # self.camera.open('rtsp://Kornfeil:Kornfeil7@10.150.10.153 /1')

    def slideshow(self):
        for imgName in glob.glob("/beta/Work/2/postbake/*.png", recursive=True):
            img = cv2.imread(imgName)
            # img = cv2.resize(img, dsize=None, fx=0.7, fy=0.7)
            img = img[:, 100:-300]
            # img = cv2.resize(img, (800, 800))
            # out = []
            out = self.P.predict(img, threshold=0.07)
            for i, x in enumerate(out):
                cid = x[0]
                score = x[1]
                roi = x[2:]

                label = '|{}|.{:.3f}'.format(cid, score)
                ImgUtils.drawRect(roi, img, colour=(255, 0, 0))
                cv2.putText(img=img, text=label, org=(int(roi[0]), int(roi[1])),
                            fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=1, lineType=cv2.LINE_4,
                            fontScale=2, color=(0, 255, 255))
            print('\n')
            while True:
                ImgUtils.show("Feed", img, 0, 0)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    return
                elif key == ord('v'):
                    break


    # def main(self):
    #     clock = 0
    #     while True:
    #         clock += 1
    #         _, self.rawFrame = self.camera.read()
    #         if self.rawFrame is None:
    #             continue
    #
    #         self.frame = StandardDetectionTrans.prepareResized(self.rawFrame)
    #         out = self.P.getBoxes(self.frame, threshold=0.1)
    #
    #         for i, x in enumerate(out):
    #             cid = x[0]
    #             score = x[1]
    #             roi = x[2:]
    #             print(cid, roi)
    #             label = '|{}|.{:.3f}'.format(cid, score)
    #             ImgUtils.drawRect(roi, self.frame, colour=(255, 0, 0))
    #             cv2.putText(img=self.frame, text=label, org=(int(roi[0]), int(roi[1])),
    #                         fontFace=cv2.FONT_HERSHEY_PLAIN,  thickness=1, lineType=cv2.LINE_4,
    #                         fontScale=2, color=(0, 255, 255))
    #
    #         # if clock > 40:
    #         #     self.P.reinit()
    #         gc.collect()
    #         ImgUtils.show("Live", self.frame, 800, 0)
    #         keyboard = cv2.waitKey(30)
    #         if keyboard == 'q' or keyboard == 27:
    #             break

    @staticmethod
    def box_to_rect(box, linewidth=1):
        return plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, linewidth=linewidth)


if __name__ == "__main__":
    m = Manager()
    m.slideshow()

    # camera = cv2.VideoCapture()
    # camera.open('rtsp://Kornfeil:Kornfeil7@10.150.10.153 /1')
    # while True:
    #     _, x = camera.read()
    #     if x is None:
    #         continue
    #     ImgUtils.show("x", x, 0, 0)
    #     key = cv2.waitKey(13)
    #     if key == ord('q'):
    #         break