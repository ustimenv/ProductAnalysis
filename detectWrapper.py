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
            (1, 0) : ('rtsp://Operator:PHZOperator@10.150.10.155/1', 7),  # raw
            (1, 1) : ('rtsp://Operator:PHZOperator@10.150.10.154/1', 7),  # postbake
            (3, 0) : ('rtsp://Operator:PHZOperator@10.150.10.156/1', 5)   # brick
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
        self.D.guiMode = guiMode

        if not self.guiMode:
            self.writer = SocketWriter(port)

        self.cameraIp, self.frameRate = self.cameras[(lineNumber, positionOnLine)]

        self.camera = cv2.VideoCapture()
        self.filenamePrefix = str(positionOnLine) + '/' + 'PassiveSamples/'

    def video(self):
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

                feed, contrast, out = self.D.detect(feed)

                if self.guiMode:
                    ImgUtils.show("Live", feed, 0, 0)
                    ImgUtils.show("Contrast", contrast, 0, 600)

                # TRANSMIT FINDINGS
                if not self.guiMode and curTime - writeTime >= self.samplingPeriod:
                    try:
                        num = self.D.numObjects-previousN
                        transmission = str(num) + '#'

                        if num > 0 and self.D.colourTracking and self.D.dimTracking:
                            size = int(self.D.averageSize / num)
                            colour = [str(int(i / num)) for i in self.D.averageColour]
                            transmission += (str(size) + '#' + str(colour[2])+'|'+str(colour[1])+'|'+str(colour[0])+'|')
                            self.D.averageColour = [0, 0, 0]
                            self.D.averageSize = 0

                        self.writer.write(transmission)
                        previousN = self.D.numObjects
                        self.writer.flush()

                    except Exception as e:
                        print("______Critical error", file=sys.stderr)
                    writeTime = curTime

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return

    def slideshow(self):
        previousN = 0
        for i in range(1, 10000):
            srcPath = '/beta/Work/2/postbake2/'+str(i)+'.png'
            img = cv2.imread(srcPath)
            if img is None:
                continue
            feed, contrast, out = self.D.detect(img)

            if i % 40 == 0:
                num = self.D.numObjects - previousN
                transmission = str(num) + '#'

                if num > 0 and self.D.colourTracking and self.D.dimTracking:
                    size = int(self.D.averageSize / num)
                    colour = [str(int(i/num)) for i in self.D.averageColour]
                    transmission += (str(size) + '#' + str(colour[2]) + '|' + str(colour[1]) + '|' + str(colour[0]) + '|')
                    self.D.averageColour = [0, 0, 0]
                    self.D.averageSize = 0

                previousN = self.D.numObjects
                print(transmission)

            while True:
                ImgUtils.show("Live", feed, 0, 0)
                ImgUtils.show("Contrast", contrast, 700, 0)
                keyboard = cv2.waitKey(1)
                if keyboard == ord('v'):
                    break
                elif keyboard == ord('q'):
                    return

    @staticmethod
    def testCamera():
        prevTime = 0
        counter = 8850
        camera = cv2.VideoCapture()
        camera.open('rtsp://Operator:PHZOperator@10.150.10.154/1')
        while True:
            _, feed = camera.read()
            if feed is None:
                continue
            timeElapsed = time.time() - prevTime
            if timeElapsed > 1.0/7:
                prevTime = time.time()
                # cv2.imwrite('/beta/Work/2/postbake2/'+str(counter)+'.png', feed)
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

    # pos = 'postbakeDebug'
    D = DetectorWrapper(lineNumber=1, positionOnLine=1, samplingPeriod=10000000, guiMode=True, port=-1)
    D.slideshow()
    # D.video()
    # DetectorWrapper.testCamera()
