import argparse
import glob
import logging
import traceback

import cv2
import numpy as np
from detectorFactory import DetectorFactory
from imgUtils import ImgUtils
import time
from networking import SocketWriter
import sys


class DetectorWrapper:
    # (line, production stage) -> (ip address)
    cameras = \
        {
            (1, 'raw'     ) : ('rtsp://Operator:PHZOperator@10.150.10.155/1', 7),
            (1, 'postbake') : ('rtsp://Operator:PHZOperator@10.150.10.154/1', 7),
            (1, 'brick') : ('rtsp://Operator:PHZOperator@10.150.10.156/1', 5)
        }

    def __init__(self, lineNumber, positionOnLine, port, samplingPeriod, guiMode=False):
        """
        :param lineNumber: production line this detector instance is looking at
        :param positionOnLine:  production stage, either raw dough or straight out of the oven
        :param samplingPeriod: time in seconds - either how often to either send or save frames
        :param guiMode: whether to transmit the data to the server
        """
        self.guiMode = guiMode                                # is the programme live, ig whether to transmite data
        self.samplingPeriod = samplingPeriod                    # number of seconds betweeen each transition

        self.D = DetectorFactory.getDetector(lineNumber, positionOnLine)

        if not self.guiMode:
            self.writer = SocketWriter(port)

        self.cameraIp, self.frameRate = self.cameras[(lineNumber, positionOnLine)]

        self.camera = cv2.VideoCapture()
        self.filenamePrefix = str(positionOnLine) + '/' + 'PassiveSamples/'

    def video(self, name):
        self.camera.open(self.cameraIp)
        writeTime = time.time()
        saveTime = time.time()
        prevTime = 0
        previousN = 0
        while True:
            sys.stderr.flush()
            sys.stdout.flush()
            timeElapsed = time.time() - prevTime
            _, feed = self.camera.read()
            if feed is None:
                continue

            if timeElapsed > 1.0/self.frameRate:        # reduce framerate to ease the stress on the camera & network
                prevTime = time.time()
                curTime = time.time()

                # TRANSMIT FINDINGS
                if not self.guiMode and curTime - writeTime >= self.samplingPeriod:
                    try:
                        self.writer.write(str(self.D.numObjects-previousN))
                        previousN = self.D.numObjects
                        self.writer.flush()
                    except:
                        print("______Critical error", file=sys.stderr)
                    writeTime = curTime

                feed, contrast = self.D.detect(feed)
                if self.guiMode:
                    ImgUtils.show("Live", feed, 0, 0)
                    # ImgUtils.show("Contrast", contrast, 0, 600)

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return

    def slideshow(self):
        for i in range(1, 9988):
            srcPath = '/beta/Work/2/raw2/'+str(i)+'.png'
            img = cv2.imread(srcPath)
            if img is None:
                continue
            feed, contrast = self.D.detect(img)

            if True:
                ImgUtils.show("Live", feed, 0, 0)
                ImgUtils.show("Contrast", contrast, 700, 0)
                keyboard = cv2.waitKey(13)
                if keyboard == ord('v'):
                    break
                elif keyboard == ord('q'):
                    return

    @staticmethod
    def testCamera():
        prevTime = 0
        counter = 1
        camera = cv2.VideoCapture()
        camera.open('rtsp://Operator:PHZOperator@10.150.10.155/1')
        while True:
            _, feed = camera.read()
            if feed is None:
                continue
            timeElapsed = time.time() - prevTime
            if timeElapsed > 1.0/7:
                prevTime = time.time()
                # cv2.imwrite('/beta/Work/2/raw2/'+str(counter)+'.png', feed)
                counter += 1
                ImgUtils.show('Img', feed, 0, 0)
            if counter > 10000:
                print("Sufficient samples")
                return
            keyboard = cv2.waitKey(30)
            if keyboard == ord('v'):
                break
            elif keyboard == ord('q'):
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--position', dest='position', type=str, default='brick')
    args = parser.parse_args()
    pos = args.position
    # pos = 'postbakeDebug'
    D = DetectorWrapper(lineNumber=1, positionOnLine=pos, samplingPeriod=10000000, guiMode=True, port=-1)
    # D.slideshow()
    D.video(pos)
    # DetectorWrapper.testCamera()
