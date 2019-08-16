import logging

import cv2
import numpy as np
from detect import Detector
from imgUtils import ImgUtils
import time
from networking import SocketWriter
import sys
INTMAX = 2**60

class DetectorWrapper:
    # (line, production stage) -> (ip address)
    cameras = \
        {
            (1, 'raw'     ) : 'rtsp://Operator:PHZOperator@10.150.10.155/1',
            (1, 'postcool') : 'rtsp://Operator:PHZOperator@10.150.10.154/1'
        }
    args = \
    {
        (1, 'raw')      : {'expectedWidth': 140, 'expectedHeight': 100,        # detection params
                            'transformationTarget': 'raw',                      # select correct image transformations
                            'upperKillzone': INTMAX, 'lowerKillzone': -INTMAX,              # select correct tracking parameters
                            'rightKillzone': INTMAX+680, 'leftKillzone': 300-INTMAX,              # select correct tracking parameters
                            'timeToDie': 5,      'timeToLive': 0,
                            'partioningRequired' : False
                    },

        (1, 'postcool') : {'expectedWidth' : 70, 'expectedHeight'     : 70,     # detection params
                           'transformationTarget'             : 'cool',  # select correct image transformations
                           'upperKillzone'  : 550, 'lowerKillzone' : 220,     # select correct tracking parameters
                           'rightKillzone'  : 3000, 'leftKillzone' : -3000,     # select correct tracking parameters
                           'timeToDie' : 9, 'timeToLive'     : 3,
                           'partioningRequired': True
                          }
    }

    def __init__(self, lineNumber, position, port, samplingRate, cameraMode=True, showFeed=False, run=True, startNum=0):
        """
        :param lineNumber: production line this detector instance is looking at
        :param position:  production stage, either raw dough or straight out of the oven
        :param samplingRate: time in seconds - either how often to either send or save frames
        :param cameraMode: whether detector is running on a series of individual frames or a video stream
        :param showFeed:    duh
        :param run: whether to transmit the data to the server
        """
        self.lineNumber = lineNumber
        self.position = position
        self.D = Detector(**self.args[(self.lineNumber, self.position)], startNum=startNum)

        self.port = port
        self.run = run
        self.samplingRate = samplingRate

        self.cameraMode = cameraMode
        self.showFeed = showFeed

        if self.run:
            self.writer = SocketWriter(self.port)

        cameraIp = self.cameras[(self.lineNumber, position)]
        self.camera = cv2.VideoCapture()
        self.camera.open(cameraIp)

    def collectImageSample(self, img, n):
        cv2.imwrite(str(n) + '.png', img)

    def video(self):
        counter = 0
        flushCounter = 0
        startTime = time.time()
        while True:
            sys.stderr.flush()
            sys.stdout.flush()

            _, feed = self.camera.read()
            if feed is None:
                continue
            counter += 1
            curTime = time.time()

            if self.run and curTime - startTime >= self.samplingRate:
                try:
                    self.writer.write(str(self.D.numObjects))
                    self.writer.flush()
                except:
                    print("______Critical error", file=sys.stderr)
                startTime = curTime

            feed = self.D.resize(feed)
            frame = self.D.getImgWithBoxes(np.copy(feed))
            # self.collectSample(feed, counter)
            # print(self.D.numObjects)
            if self.showFeed:
                if self.position == "raw":
                    xPos = 0
                    yPos = 0
                else:
                    xPos = 1600
                    yPos = 0
                ImgUtils.show("Live"+str(self.position), frame, xPos, yPos)


                # X = self.D.transformer.transform(feed)
                # ImgUtils.show("Contrast", X, 800, 00)
                # ImgUtils.show("Live"+str(self.position), frame, 0, 0)
                # ImgUtils.show("Contrast", X, 0, 500)

            keyboard = cv2.waitKey(30)
            if keyboard == 27:
                break
            elif keyboard == ord('q'):
                return

        return self.D.tracker.N

    def slideshow(self):
        for i in range(1, 560):
            img = cv2.imread('raw/4/' + str(i) + '.png')
            img = self.D.transformer.transformResize(img)
            contrast = self.D.transform(np.copy(img))
            img = self.D.getImgWithBoxes(img)

            while True:
                ImgUtils.show('Img', img, 0, 0)
                ImgUtils.show('Contrast', contrast, 1000, 0)

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return

# logging.basicConfig(filename="/home/vlad/pylog.log", format='%(asctime)s %(message)s', filemode='w')

if __name__ == "__main__":
    D = DetectorWrapper(lineNumber=1, position='postcool', showFeed=True, samplingRate=10000000, run=False, port=-1, startNum=0)
    D.video()
    # D.slideshow()
