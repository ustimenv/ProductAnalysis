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
            (1, 'postbake') : 'rtsp://Operator:PHZOperator@10.150.10.154/1',
            (1, 'postbakeDebug') : 'rtsp://Operator:PHZOperator@10.150.10.153/1'
        }
    args = \
    {
        (1, 'raw')      : {'expectedWidth': 180, 'expectedHeight': 130,        # detection params
                            'target': 'raw',                      # select correct image transformations
                            'upperKillzone': INTMAX, 'lowerKillzone': -INTMAX,              # select correct tracking parameters
                            'rightKillzone': INTMAX+680, 'leftKillzone': 300-INTMAX,              # select correct tracking parameters
                            'timeToDie': 1,      'timeToLive': 0,
                            'roiTrackingMode' : True,
                    },

        (1, 'postbake') : {'expectedWidth' : 120, 'expectedHeight'     : 120,     # detection params
                           'target'        : 'postbake',        # select correct image transformations
                           'upperKillzone'  : 550, 'lowerKillzone' : 220,     # select correct tracking parameters
                           'rightKillzone'  : 3000, 'leftKillzone' : -3000,     # select correct tracking parameters
                           'timeToDie' : 2, 'timeToLive'     : 0,
                           'roiTrackingMode' : False,
                           },

        (1, 'postbakeDebug') : {'expectedWidth' : 120, 'expectedHeight'     : 120,     # detection params
                           'target'        : 'postbake',        # select correct image transformations
                           'upperKillzone'  : 550, 'lowerKillzone' : 220,     # select correct tracking parameters
                           'rightKillzone'  : 3000, 'leftKillzone' : -3000,     # select correct tracking parameters
                           'timeToDie' : 1, 'timeToLive'     : 0,
                           'roiTrackingMode' : False
                          }
    }

    def __init__(self, lineNumber, position, port, samplingPeriod, cameraMode=True, guiMode=False):
        """
        :param lineNumber: production line this detector instance is looking at
        :param position:  production stage, either raw dough or straight out of the oven
        :param samplingPeriod: time in seconds - either how often to either send or save frames
        :param cameraMode: whether detector is running on a series of individual frames or a video stream
        :param showFeed:    duh
        :param guiMode: whether to transmit the data to the server
        """
        self.D = Detector(**self.args[(lineNumber, position)])  # internal detector
        self.guiMode = guiMode                                # is the programme live, ig whether to transmite data
        self.samplingPeriod = samplingPeriod                    # number of seconds betweeen each transition

        self.cameraMode = cameraMode

        if not self.guiMode:
            self.writer = SocketWriter(port)

        cameraIp = self.cameras[(lineNumber, position)]
        self.camera = cv2.VideoCapture()
        self.camera.open(cameraIp)

        self.frameRate = 10

    def collectImageSample(self, img, n):
        cv2.imwrite('raw/1/'+str(n) + '.png', img)

    def video(self):
        counter = 0
        writeTime = time.time()
        saveTime = time.time()
        prevTime = 0
        while True:
            sys.stderr.flush()
            sys.stdout.flush()

            timeElapsed = time.time() - prevTime
            _, feed = self.camera.read()
            if feed is None:
                continue

            if timeElapsed > 1.0/self.frameRate:
                prevTime = time.time()
                curTime = time.time()

                if self.guiMode and curTime - writeTime >= self.samplingPeriod:
                    try:
                        self.writer.write(str(self.D.numObjects))
                        self.D.numObjects = 0
                        self.writer.flush()
                    except:
                        print("______Critical error", file=sys.stderr)
                    writeTime = curTime
                if time.time() - saveTime >= 600 or True: #600=save every 10 minutes
                    counter += 1
                    # self.collectImageSample(feed, counter)
                    # cv2.imwrite("Samples/"+str(self.position) + "|" + str(int(time.time())) + '.png', feed)
                    saveTime = time.time()

                feed = self.D.resize(feed)
                frame = self.D.getImgWithBoxes(np.copy(feed))

                if self.guiMode:
                    ImgUtils.show("Live", frame, 0, 0)
                    X = self.D.transformer.transform(feed)
                    ImgUtils.show("Contrast", X, 0, 600)

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return



    def slideshow(self):
        for i in range(1, 1280):
            img = cv2.imread('raw/1/' + str(i) + '.png')
            if img is None:
                continue
            # img = self.D.transformer.transformResize(img)
            contrast = self.D.transform(np.copy(img))
            # img = self.D.getImgWithBoxes(img)

            img = self.getBeltCoordinates(img)

            while True:
                ImgUtils.show('Img', img, 0, 0)
                # ImgUtils.show('Contrast', contrast, 00, 500)

                keyboard = cv2.waitKey(30)
                if keyboard == 27:
                    break
                elif keyboard == ord('q'):
                    return

    def _meanLines(self, lines):
        meanCoords = [0, 0, 0, 0]  # xmin, ymin, xmax, ymax

        _meanX = 0
        _meanY = 0

        # firstly find the mean x- and y-intercepts
        if lines is not None and len(lines) > 0:
            coordinates = []
            for line in lines:
                rho = line[0][0]; theta = line[0][1]
                a = np.math.cos(theta)
                b = np.math.sin(theta)
                x = a * rho;   y = b * rho
                _meanX += x;    _meanY += y
                coordinates.append([x, y])

            _meanX /= len(lines);        _meanY /= len(lines)

            meanCoordsCounter = [0, 0, 0, 0]    # num <X, num <Y, num >= X, num >= Y
            for x, y in coordinates:
                if x < _meanX:
                    meanCoords[0] += x;   meanCoordsCounter[0] += 1
                else:
                    meanCoords[2] += x; meanCoordsCounter[2] += 1
                if y < _meanY:
                    meanCoords[1] += y; meanCoordsCounter[1] += 1
                else:
                    meanCoords[3] += y; meanCoordsCounter[3] += 1

            for i in range(len(meanCoords)):
                meanCoords[i] = int(meanCoords[i] / meanCoordsCounter[i])

        return meanCoords


    def getBeltCoordinates(self, img):
        roi = [0 ,0, 0, 0]       # xmin, ymin, xmax, ymax
        gray = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2GRAY)
        height, width, _= img.shape
        edges = cv2.Canny(gray, 200, 600)

        linesHorizontal = cv2.HoughLines(image=edges, rho=1, theta=(np.pi/180), threshold=150,
                                         min_theta=45*(np.pi/180), max_theta=90*(np.pi/180))   #horizontal

        _roi = self._meanLines(linesHorizontal)
        roi[1], roi[3] = _roi[1],  _roi[3]

        linesVertical = cv2.HoughLines(image=edges, rho=1, theta=(np.pi/180), threshold=150,
                                       min_theta=-45*(np.pi/180), max_theta=45*(np.pi/180))     # vertical
        _roi = self._meanLines(linesVertical)
        roi[0], roi[2] = _roi[0], _roi[2]
        roi[0]-=150; roi[2]-=150
        ImgUtils.drawRect(roi, img)
        return img


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
    D = DetectorWrapper(lineNumber=1, position='postbake', samplingPeriod=10000000, guiMode=True, port=-1)
    # D.testCamera()
    # D.video()
    D.slideshow()
