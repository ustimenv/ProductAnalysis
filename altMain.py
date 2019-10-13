import glob
import cv2
from imgUtils import ImgUtils
from altPredict import Predictor
import matplotlib.pyplot as plt


class Manager:
    def __init__(self):
        self.P = Predictor()
        self.camera = cv2.VideoCapture()

    def pad(self, img, newDim=1024):
        h, w, _ = img.shape

        gmi = cv2.copyMakeBorder(src=img, top=0, bottom=newDim - h, left=0, right=newDim - w,
                                 borderType=cv2.BORDER_REFLECT)
        return gmi

    def test(self):
        for imgName in glob.glob("/beta/Work/2/postbake/*.png", recursive=True):
        # for imgName in glob.glob("/beta/Work/2/Train/1/*.png", recursive=True):
            img = cv2.imread(imgName)
            # img = cv2.resize(img, dsize=None, fx=0.7, fy=0.7)
            img = img[50:-50, 500:850]
            # img = cv2.resize(img, (300, 300))
            # out = []
            out = self.P.predict(img, threshold=0.7)
            for i, x in enumerate(out):
                print(x)
                cid = x[0]
                score = x[1]
                roi = x[2]

                centre = ImgUtils.getCentroid(roi)
                cv2.circle(img, centre, 20, (255, 0, 255), 3)

                label = '|{}|.{:.3f}'.format(cid, score)
                ImgUtils.drawRect(roi, img, colour=(255, 0, 0))
                cv2.putText(img=img, text=label, org=(int(roi[0]), int(roi[1])),
                            fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=1, lineType=cv2.LINE_4,
                            fontScale=2, color=(0, 255, 255))

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
    m.test()

