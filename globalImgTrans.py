import glob
import os

from PIL import Image
from PIL.ImagePath import Path

from ohtaTrans import StandardDetectionTrans
from imgUtils import *


class GlobImgTrans:

    @staticmethod
    def averageImageSize():
        xMax = -1
        yMax = -1
        for _ in os.walk('Data/Train/'):
            totalX = 0
            totalY = 0
            totalNum = 0
            curDir = _[0]
            for imName in Path(curDir).glob('**/*.png'):
                image = Image.open(imName)
                totalX += image.size[0]
                totalY += image.size[1]
                totalNum += 1
            averageX = int(totalX / totalNum)
            averageY = int(totalY / totalNum)
            if averageX > xMax:
                xMax = averageX
            if averageY > yMax:
                yMax = averageY
        return xMax + 30, yMax + 30

    @staticmethod
    def resizeToAverage(dims, path='.'):
        for imName in Path(path).glob('**/*.png'):
            fullName = './' + str(imName)
            image = cv2.imread(str(imName))
            resized = cv2.resize(image, dsize=dims)
            cv2.imwrite(fullName, resized)

    @staticmethod
    def resizeGlobal():

        for imName in glob.glob('**/*.png', recursive=True):
            StandardDetectionTrans.resizeCustom(imName)


    @staticmethod
    def padGlob(padDims=300):
        for _ in os.walk('Data/Train/'):
            totalX = 0
            totalY = 0
            totalNum = 0
            curDir = _[0]
            for imName in Path(curDir).glob('**/*.png'):
                image = Image.open(imName)
                totalX += image.size[0]
                totalY += image.size[1]
                totalNum += 1
            averageX = int(totalX / totalNum)
            averageY = int(totalY / totalNum)
            if averageX > xMax:
                xMax = averageX
            if averageY > yMax:
                yMax = averageY


    #to be used stirclty for contour extraction at dataset preparation stage!!
    @staticmethod
    def averageRoi(imgName):
        rois = GlobImgTrans.detect(imgName)
        roiTotal = [0, 0, 0, 0]
        if len(rois) == 0:
            return (0, 0, 0, 0)
        for roi in rois:
            for i in range(4):
                roiTotal[i]+=roi[i]
        return [r / len(rois) for r in roiTotal]

    @staticmethod
    def detect(imgName):
        img = cv2.imread(imgName)
        contrast = StandardDetectionTrans.prepareRefinedMono(img)

        contours, h = cv2.findContours(contrast, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        currentRois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) == 0 or w < 60 or h < 60 or h > w:
                continue
            # coordinates of the current bounding box
            x1 = x #- 40
            x2 = x1 + w# + 60
            y1 = y #- 20
            y2 = y1 + h #+ 50
            currentRois.append((x1, y1, x2, y2))
        return currentRois

    @staticmethod
    def averageRoiSize():
        for outerDirs in glob.glob('Data/Train/*', recursive=False):
            roisByDir= {outerDirs.split('/')[2] : [0, 0, 0, 0]}
            for fullPath in glob.glob('Data/Train/**/*.png', recursive=True):
                category = fullPath.split('/')[2]
                roi = GlobImgTrans.averageRoi(fullPath)
                if category in roisByDir.keys():
                    for i in range(4):
                        roisByDir[category][i]+=roi[i]
                else:   #failsafe
                    roisByDir[category]=roi
