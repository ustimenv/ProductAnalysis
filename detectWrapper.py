import cv2
import numpy as np
from detect import Detector
from imgUtils import ImgUtils
import time
from networking import SocketWriter


class DetectorWrapper:
    # (line, production stage) -> (ip address)
    cameras = \
        {
            (1, 'raw'     ) : 'rtsp://Operator:PHZOperator@10.150.10.155/1',
            (1, 'postcool') : 'rtsp://Operator:PHZOperator@10.150.10.154/1'
        }
    args = \
    {
        (1, 'raw')      : {'expectedW' : 130, 'expectedH'     : 120,    # detection params
                           'wLow'      : 100,  'hLow'          : 90,
                           'transformationTarget'             : 'raw',  # select correct image transformations
                           'killzone'  : 600, 'lowerKillzone' : 200,    # select correct tracking parameters
                           'timeToDie' : 4, 'timeToLive'      : 4,
                           'movementDirection'                : -1
                          },

        (1, 'postcool') : {'expectedW' : 300, 'expectedH'     : 300,     # detection params
                           'wLow'      : 80,  'hLow'          : 80,
                           'transformationTarget'             : 'cool',  # select correct image transformations
                           'killzone'  : 550, 'lowerKillzone' : 220,     # select correct tracking parameters
                           'timeToDie' : 13, 'timeToLive'     : 13,
                           'movementDirection'                : 1
                          }
    }

    def __init__(self, lineNumber, position, port, samplingRate, cameraMode=True, showFeed=False):
        """

        :param lineNumber: production line this detector instance is looking at
        :param position:  production stage, either raw dough or straight out of the oven
        :param samplingRate: time in seconds - either how often to save frames or how often to send the data to server
        :param cameraMode: whether detector is running on a series of individual frames or a video stream
        :param showFeed:    duh
        """
        self.lineNumber = lineNumber
        self.position = position
        self.port = port

        self.D = Detector(**self.args[(self.lineNumber, self.position)])
        self.cameraMode = cameraMode
        self.showFeed = showFeed
        self.samplingRate = samplingRate
        # self.writer = SocketWriter(destinationPort=50000 + self.lineNumber)
        self.writer = SocketWriter(self.port)

        if self.cameraMode:
            cameraIp = self.cameras[(self.lineNumber, position)]
            self.camera = cv2.VideoCapture()
            self.camera.open(cameraIp)

    def video(self, sample=False):
        if not self.cameraMode:
            raise Exception
        counter = 0
        s = 0
        startTime = time.time()
        while True:
            curTime = time.time()
            _, feed = self.camera.read()
            print(self.position, self.D.tracker.N)
            if curTime - startTime >= self.samplingRate:
                # print("SENding")
                self.writer.send(str(self.D.numObjects))
                startTime = curTime
            if feed is None:
                continue
            feed = self.D.resize(feed)
            contr = self.D.transform(np.copy(feed))
            frame = self.D.getImgWithBoxes(np.copy(feed))

            counter += 1
            if sample:
                if counter % self.samplingRate == 0:
                    s += 1
                    cv2.imwrite(str(s)+'.png', feed)

            if self.showFeed:
                # ImgUtils.show("1", contr, 800, 0)
                if self.position == "raw":
                    loc = 1200
                else:
                    loc = 1600
                ImgUtils.show("Live"+str(self.position), frame, loc, 0)
                # ImgUtils.show("With boxes", frame, 1600, 0)
                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return

        return self.D.tracker.N

    def slideshow(self):
        for i in range(1, 348):
            img = cv2.imread('pstcool/2/' + str(i) + '.png')
            contrast = self.D.transform(np.copy(img))
            img = self.D.getImgWithBoxes(img)

            while True:
                ImgUtils.show('Img', img, 1100, 0)
                ImgUtils.show('Contrast', contrast, 1500, 0)

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return


if __name__ == "__main__":
    D = DetectorWrapper(lineNumber=1, position='raw', showFeed=True, samplingRate=10000000)
    D.video()
