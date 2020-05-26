import argparse
import glob
import logging
import traceback
from collections import deque

import cv2
import numpy as np
from miscellaneous import Constants
from detectorFactory import DetectorFactory
from utils.imgUtils import ImgUtils
import time
from utils.networking import SocketWriter
import sys

class DetectorWrapper:

    def __init__(self, lineNumber, positionOnLine, port, samplingPeriod, debugMode=False):
        """
        :param lineNumber: production line this detector instance is looking at
        :param positionOnLine:  production stage, either raw dough or straight out of the oven
        :param samplingPeriod: time in seconds - either how often to either send or save frames
        :param debugMode: if true, do not transmit data to the server
        """

        self.D = DetectorFactory.getDetector(lineNumber, positionOnLine)
        self.debugMode = debugMode
        self.samplingPeriod = samplingPeriod  # rate of writing data to the server

        self.D.guiMode = debugMode  # TODO delete

        if not self.debugMode:
            self.writer = SocketWriter(port)

        cameraIp = Constants.cameras[(lineNumber, positionOnLine)]
        self.camera = cv2.VideoCapture()
        self.camera.open(cameraIp)

    def video(self):
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

            if timeElapsed > 1.0 / 6:  # reduce frame rate to ease the stress on the camera & network
                prevTime = time.time()
                curTime = time.time()

                feed, contrast, out = self.D.detect(feed)

                if self.debugMode:
                    ImgUtils.show("Live", feed, 0, 0)
                    ImgUtils.show("Contrast", contrast, 0, 600)

                # transmit our findings
                if curTime - writeTime >= self.samplingPeriod:
                    num = self.D.numObjects - previousN
                    if not self.debugMode:
                        try:
                            transmission = str(num) + '#'
                            self.writer.write(transmission)
                            self.writer.flush()

                        except Exception as e:
                            print("______Critical error", file=sys.stderr)
                        writeTime = curTime
                    else:
                        print('Counted ', str(num), ' in ', str(self.samplingPeriod), ' frames')

                    previousN = self.D.numObjects

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return

    def slideshow(self, srcPath):
        previousN = 0
        for i in range(1, 10000):
            srcPath += str(i) + '.png'
            img = cv2.imread(srcPath)
            if img is None:
                continue
            feed, contrast, out = self.D.detect(img)

            if i % self.samplingPeriod == 0:
                num = self.D.numObjects - previousN
                previousN = self.D.numObjects
                print(num)
            while True:
                ImgUtils.show("Live", feed, 0, 0)
                ImgUtils.show("Contrast", contrast, 700, 0)
                keyboard = cv2.waitKey(1)
                if keyboard == ord('q'):
                    return
                if keyboard == ord('v'):
                    break


