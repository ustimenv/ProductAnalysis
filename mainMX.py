import cv2
import matplotlib.pyplot as plt

from imgUtils import ImgUtils
from predict import *


class Manager:
    camera = cv2.VideoCapture()
    camera2 = cv2.VideoCapture()
    D = StandardDetector()
    P = Predictor()
    rawFrame = None
    frame = None
    rois = None

    def __init__(self):
        self.camera.open('rtsp://operator:Operator@10.110.1.55/1')
        # self.camera.open('rtsp://Operator:PHZOperator@10.150.10.154/1')
        # self.camera2.open('rtsp://Operator:PHZOperator@10.150.10.155/1')
        # self.camera.open('rtsp://Kornfeil:Kornfeil7@10.150.10.153 /1')

    def main(self):
        clock = 0
        while True:
            clock += 1
            _, self.rawFrame = self.camera.read()
            if self.rawFrame is None:
                continue

            self.frame = StandardDetectionTrans.prepareResized(self.rawFrame)
            out = self.P.getBoxes(self.frame, threshold=0.1)

            for i, x in enumerate(out):
                cid = x[0]
                score = x[1]
                roi = x[2:]
                print(cid, roi)
                label = '|{}|.{:.3f}'.format(cid, score)
                ImgUtils.drawRect(roi, self.frame, colour=(255, 0, 0))
                cv2.putText(img=self.frame, text=label, org=(int(roi[0]), int(roi[1])),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,  thickness=1, lineType=cv2.LINE_4,
                            fontScale=2, color=(0, 255, 255))

            # if clock > 40:
            #     self.P.reinit()
            gc.collect()
            ImgUtils.show("Live", self.frame, 800, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    def showFeed(self):
        while True:
            _, feed = self.camera.read()
            if feed is None:
                continue
            ImgUtils.show("Feed", feed, 0, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    @staticmethod
    def box_to_rect(box, linewidth=1):
        return plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, linewidth=linewidth)

    def test(self):
        for i in range(0, 100):
            name = '../samples/'+str(i)+'.png'
            img = cv2.imread(name)
            if img is None:
                continue

            plt.imshow(img)
            out = self.P.getBoxes(img, 0.4)
            print('\n', name[-7:-5])
            topTen = 0
            for x in out:
                cid = x[0]
                topTen+=1
                score = x[1]
                roi = x[2:]
                plt.gca().add_patch(Manager.box_to_rect(roi))
                if topTen < 4:
                    label = '|{}|.{:.3f}'.format(cid, score)
                    plt.text(roi[0], roi[1], label)
                    print(cid , '===', score)
            plt.show()


    def test2(self):
        for i in range(1, 3):
            name = 'feed'+str(i)+'.png'
            img = cv2.imread(name)
            if img is None:
                continue

            plt.imshow(img)
            out = self.P.getBoxes(img, 0.0)
            print(name)
            for x in out:
                cid = x[0]
                if cid == -1:
                    continue
                score = x[1]
                roi = x[2:]
                plt.gca().add_patch(Manager.box_to_rect(roi))
                label = '|{}|.{:.3f}'.format(cid, score)
                plt.text(roi[0], roi[1], label)
                print(x)
            plt.show()

    def prepareTestData(self):
        for i in range(0, 100):
            name = '../samples/'+str(i)+'.png'
            img = cv2.imread(name)
            if img is None:
                continue

            StandardDetectionTrans.flip4Way(name, '../samples/F' + str(i) + '.png')



if __name__ == "__main__":
    m = Manager()
    m.main()