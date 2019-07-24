from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from detector import Detector
from imageTrans import ImageTransforms
from predict import *
from sampler import Sampler
from utils import ImgUtils


class Manager:
    camera = cv2.VideoCapture()
    D = Detector()
    S = Sampler()
    P = Predictor()

    liveFeed = None
    frame = None
    rois = None

    def __init__(self):
        self.camera.open('rtsp://operator:Operator@10.110.1.55/1')
        self.sampling_rate = 4
        # self.camera.open('rtsp://Kornfeil:Kornfeil7@10.150.10.153 /1')

    def main(self):
        clock = 0
        while True:
            clock += 1
            _, self.liveFeed = self.camera.read()
            if self.liveFeed is None:
                continue

            self.frame = ImageTransforms.prepareResized(self.liveFeed)
            contrast = ImageTransforms.prepareMono(self.liveFeed)

            if clock % self.sampling_rate == 0:
                pass
                # self._sample()

            ImgUtils.show("Frame", contrast, 0, 0)

            if clock % 4<0:
                try:
                    out = self.P.getBoxes(self.frame, threshold=0)
                except:
                    print('top kek')

                for i, x in enumerate(out):
                    if i>=6:
                        break
                    cid = x[0]
                    if cid == 8:
                        continue

                    score = x[1]
                    roi = x[2:]
                    label = '|{}|.{:.3f}'.format(cid, score)
                    ImgUtils.drawRect(roi, self.frame, colour=(255, 0, 0))
                    cv2.putText(img=self.frame, text=label, org=(int(roi[0]), int(roi[1])),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,  thickness=1, lineType=cv2.LINE_4,
                                fontScale=2, color=(0, 255, 255))

            else:
                rois = self.D.detect(contrast)
                for roi in rois:
                    ImgUtils.drawRect(roi, self.frame, colour=(0, 255, 255))

            gc.collect()
            ImgUtils.show("Live", self.frame, 800, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break


    def _sample(self):
        yScale = self.liveFeed.shape[0] - self.frame.shape[0]
        xScale = self.liveFeed.shape[1] - self.frame.shape[1]
        for (xmin, ymin, xmax, ymax) in self.rois:
            ymin += int(yScale / 2)
            ymax += int(yScale / 2)
            xmin += int(xScale / 2)
            xmax += int(xScale / 2)
            self.S.extract(self.liveFeed[ymin:ymax, xmin:xmax])
            # ImgUtils.drawRect((xmin, ymin, xmax, ymax), liveFeed, (255, 0, 255))

    @staticmethod
    def box_to_rect(box, linewidth=1):
        return plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             fill=False, linewidth=linewidth)


    def test(self):
        for i in range(0, 10):
            name = 'samples/'+str(i)+'X.png'
            img = cv2.imread(name)
            plt.imshow(img)
            out = self.P.getBoxes(img, 0.0)
            print(name)
            topTen = 0
            for x in out:
                topTen+=1
                cid = x[0]
                score = x[1]
                roi = x[2:]
                plt.gca().add_patch(Manager.box_to_rect(roi))
                if topTen < 10:
                    label = '|{}|.{:.3f}'.format(cid, score)
                    plt.text(roi[0], roi[1], label)
                    print(x)
            plt.show()



class Tracker:
    def __init__(self):
        pass



if __name__ == "__main__":
    # print(Janet(['1', '2', '3', '4', '5', '6', '7', '8', '9'], mx.gpu()))
    # print('\n\n\n\n\n\n\n\n\n\n')
    m = Manager()
    # m.test()
    m.main()
    # print('GPUS', num_gpus())
    # for i in range(100):
    #     print(cpu(i))

    # ImageTransforms.flip4Way('DataReduced/3X.png', 'samples/skrrt.png')