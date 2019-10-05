import glob
import logging

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
            (1, 'raw'     ) : 'rtsp://Operator:PHZOperator@10.150.10.155/1',
            (1, 'postbake') : 'rtsp://Operator:PHZOperator@10.150.10.154/1',
            (1, 'postbakeDebug') : 'rtsp://Operator:PHZOperator@10.150.10.153/1'
        }

    def __init__(self, lineNumber, positionOnLine, port, samplingPeriod, frameRate=20, guiMode=False):
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

        cameraIp = self.cameras[(lineNumber, positionOnLine)]
        self.camera = cv2.VideoCapture()
        self.camera.open(cameraIp)
        self.filenamePrefix = str(positionOnLine) + '/' + 'PassiveSamples/'
        self.frameRate = frameRate


    def video(self, name):
        counter = 0
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
                # try:
                #     f = '/home/vlad/Work/1/'+name+'/PassiveSamples/'+str(counter)+'.png'
                #     cv2.imwrite(filename=f, img=feed)
                #     counter+=1
                # except Exception:
                #     pass
                if not self.guiMode and curTime - writeTime >= self.samplingPeriod:
                    try:
                        self.writer.write(str(self.D.numObjects-previousN))
                        previousN = self.D.numObjects
                        self.writer.flush()
                    except:
                        print("______Critical error", file=sys.stderr)
                    writeTime = curTime
                if time.time() - saveTime >= 300: # 300=save every 5 minutes
                    counter += 1
                    try:
                        filename = self.filenamePrefix + \
                                   time.asctime()[3:-8].replace(' ', '|').replace(':', '|') +\
                                   '.png'
                        cv2.imwrite(filename=filename, img=feed)
                    except:
                        print("Failed to write")
                    saveTime = time.time()

                feed, contrast = self.D.detect(feed)
                if self.guiMode:
                    ImgUtils.show("Live", feed, 0, 0)
                    ImgUtils.show("Contrast", contrast, 0, 600)

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return

    @staticmethod
    def slideshow(pos):
        x = 0
        y = 0
        z = 0
        #HWC
        for filename in glob.glob("/home/vlad/Work/1/"+pos+"/UnitSamples/**.png"):
            # img, contrast = self.D.detect(img)
            img = cv2.imread(filename=filename)
            if img is None:
                continue
            if img.shape[0]<140:
                y+=1
            if img.shape[1]<140:
                x+=1
            if img.shape[0]<140 and img.shape[1]<140:
                z+=1
            # while True:
                # ImgUtils.show('Img', img, 0, 0)
                # ImgUtils.show('Contrast', contrast, 00, 500)

                # keyboard = cv2.waitKey(30)
                # if keyboard == 27:
                #     break
                # elif keyboard == ord('q'):
                #     return
        print(x, y, z)

    def testCamera(self):
        prevTime = 0
        counter = 1
        while True:
            _, feed = self.camera.read()
            if feed is None:
                continue
            timeElapsed = time.time() - prevTime
            _, feed = self.camera.read()
            if feed is None:
                continue
            if timeElapsed > 1.0/self.frameRate:
                prevTime = time.time()
                cv2.imwrite('postbake/'+str(counter)+'.png', feed)
                counter += 1

                ImgUtils.show('Img', feed, 0, 0)
            keyboard = cv2.waitKey(30)
            if keyboard == 27:
                break
            elif keyboard == ord('q'):
                return


if __name__ == "__main__":
    pos = 'postbake'
    # D = DetectorWrapper(lineNumber=1, positionOnLine=pos, samplingPeriod=10000000, guiMode=True, port=-1)
    # D.video(pos)
    # D.testCamera()
    DetectorWrapper.slideshow(pos)
