import glob
import os
import cv2
import sys
import numpy as np
from PIL import Image
from PIL.ImagePath import Path

from utils import *
from gluoncv.data import batchify

class ImageTransforms:
    @staticmethod
    def prepareMono(img):
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        contrast = ImageTransforms.prepareResized(img)
        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
        contrast = cv2.GaussianBlur(contrast, (15, 15), 0)
        contrast = cv2.adaptiveThreshold(contrast[:, :, 2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        contrast = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, morphKernel, iterations=1)
        contrast = cv2.bitwise_not(contrast)
        return contrast

    # @staticmethod
    # def prepareMono(img):
    #     morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    #     contrast = ImageTransforms.prepareResized(img)
    #     contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
    #     contrast = cv2.GaussianBlur(contrast, (15, 15), 0)
    #     c2 = cv2.adaptiveThreshold(contrast[:, :, 2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    #     c2= cv2.morphologyEx(c2, cv2.MORPH_OPEN, morphKernel, iterations=1)
    #     contrast = cv2.bitwise_not(c2)
    #     return contrast


    @staticmethod
    def prepareRefinedMono(img):
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        contrast = cv2.GaussianBlur(contrast, (15, 15), 0)
        c2 = cv2.adaptiveThreshold(contrast[:, :, 2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        c2 = cv2.morphologyEx(c2, cv2.MORPH_OPEN, morphKernel, iterations=1)
        contrast = cv2.bitwise_not(c2)
        return contrast


    @staticmethod
    def prepareResized(img):
        # return np.copy(img[150:460, 350:-300])
        return np.copy(img[150:460, 250:-240])


    @staticmethod
    def prepareMorph(img):
        morphKernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        morphKernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        newImg = cv2.morphologyEx(img, cv2.MORPH_DILATE, morphKernelY, iterations=3)
        newImg = cv2.morphologyEx(newImg, cv2.MORPH_ERODE, morphKernelX, iterations=2)
        newImg = cv2.morphologyEx(newImg, cv2.MORPH_CLOSE, morphKernelX, iterations=3)
        return newImg

    @staticmethod
    def prepareBlur(img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    @staticmethod
    def prepareContrast(img):
        contrast = cv2.threshold(img[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return contrast

    @staticmethod
    def prepare(img):
        newImg = ImageTransforms.prepareResized(img)
        newImg = ImageTransforms.sobelTransform(newImg)
        newImg = ImageTransforms.prepareBlur(newImg)
        newImg = ImageTransforms.prepareContrast(newImg)
        newImg = ImageTransforms.prepareMorph(newImg)
        return newImg

    @staticmethod
    def sobelTransform(img, absolute=True):
        return np.bitwise_or(ImageTransforms.sobelTransformX(img, absolute), ImageTransforms.sobelTransformY(img, absolute))


    @staticmethod
    def sobelTransformX(img, absolute=True):
        if absolute:
            return np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)))
        else:
            return cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

    @staticmethod
    def sobelTransformY(img, absolute=True):
        if absolute:
            return np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)))
        else:
            return cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)


    @staticmethod
    def laplaceTransform(img):
        return cv2.Laplacian(img, cv2.CV_64F)

    @staticmethod
    def resizeCustom(imgName):

        xTarget = 300
        yTarget = 300
        img = cv2.imread(imgName)
        if img is None:
            print("Failed to open ", imgName)
            return

        yl, xl, _ = img.shape
        if xl==xTarget and yl==yTarget:
            return

        if img.shape[1] < xTarget:
            print('yee')
            img = cv2.copyMakeBorder(img, 0, 0, int((xTarget - xl) / 2), int((xTarget - xl) / 2), cv2.BORDER_REPLICATE)
        if img.shape[0] < yTarget:
            print('yt')
            img = cv2.copyMakeBorder(img, int((yTarget - yl) / 2), int((yTarget - yl) / 2), 0, 0, cv2.BORDER_REPLICATE)

        img = cv2.resize(img, (300, yTarget))
        cv2.imwrite(imgName, img)

    @staticmethod
    def flip4Way(origImgName, newImgName):
        img = cv2.imread(origImgName)
        newImg = cv2.copyMakeBorder(img, 200, 200, 200, 250, cv2.BORDER_REFLECT)
        newImg = cv2.copyMakeBorder(newImg, 200, 200, 200, 250, cv2.BORDER_REPLICATE)
        cv2.imwrite(newImgName, newImg)


class GlobalImageTransforms:
    @staticmethod
    def averageImageSize():
        # images = glob.glob("Normal/Train/Novomosk/*.png")
        xMax = -1
        yMax = -1
        for _ in os.walk('Data/Train/'):
            totalX = 0
            totalY = 0
            totalNum = 0
            curDir = _[0]
            # print(curDir)
            for imName in Path(curDir).glob('**/*.png'):
                image = Image.open(imName)
                totalX += image.size[0]
                totalY += image.size[1]
                totalNum += 1
            averageX = int(totalX / totalNum)
            averageY = int(totalY / totalNum)
            # print(averageX, averageY, totalNum)
            if averageX > xMax:
                xMax = averageX
            if averageY > yMax:
                yMax = averageY

        return (xMax + 30, yMax + 30)

    @staticmethod
    def resizeToAverage(dims, path='.'):
        # print(dims)
        for imName in Path(path).glob('**/*.png'):
            fullName = './' + str(imName)
            image = cv2.imread(str(imName))
            resized = cv2.resize(image, dsize=dims)
            # cv2.imshow('then', image)
            cv2.imwrite(fullName, resized)

    @staticmethod
    def resizeGlobal():
        # for imName in Path('Data/Train/').glob('**/*.png'):
        #     fullName = './' + str(imName)
        #     print(fullName)
            # ImageTransforms.resizeCustom(fullName)

        for imName in glob.glob('**/*.png', recursive=True):
            ImageTransforms.resizeCustom(imName)


    @staticmethod
    def padGlob(padDims=300):
        for _ in os.walk('Data/Train/'):
            totalX = 0
            totalY = 0
            totalNum = 0
            curDir = _[0]
            # print(curDir)
            for imName in Path(curDir).glob('**/*.png'):
                image = Image.open(imName)
                totalX += image.size[0]
                totalY += image.size[1]
                totalNum += 1
            averageX = int(totalX / totalNum)
            averageY = int(totalY / totalNum)
            # print(averageX, averageY, totalNum)
            if averageX > xMax:
                xMax = averageX
            if averageY > yMax:
                yMax = averageY


    #to be used stirclty for contour extraction at dataset preparation stage!!
    @staticmethod
    def averageRoi(imgName):
        rois = GlobalImageTransforms.detect(imgName)
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
        contrast = ImageTransforms.prepareRefinedMono(img)

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
                roi = GlobalImageTransforms.averageRoi(fullPath)
                if category in roisByDir.keys():
                    for i in range(4):
                        roisByDir[category][i]+=roi[i]
                else:   #failsafe
                    roisByDir[category]=roi

if __name__ == "__main__":
    # GlobalImageTransforms.resizeGlobal()
    # GlobalImageTransforms.averageRoiSize()

    ImageTransforms.flip4Way('DataReduced/3X.png')
    # xTarget = 300
    # yTarget = 200
    #
    # a = cv2.imread("1img.png")
    # # xPad, yPad = GlobalImageTransforms.averageImageSize()
    # b = cv2.imread("2img.png")
    # # print(a.shape, b.shape)
    # yl, xl, _ = a.shape
    # print(a.shape)
    # # x = cv2.copyMakeBorder(a, int((yTarget-yl)/4), int((yTarget-yl)/4),
    # #                        int((xTarget - xl) / 4), int((xTarget-xl)/4),    cv2.BORDER_REPLICATE)
    #
    # img = a
    # if img.shape[1] < xTarget:
    #     print('yee')
    #     img = cv2.copyMakeBorder(img, 0, 0, int((xTarget - xl) / 2), int((xTarget - xl) / 2), cv2.BORDER_REPLICATE)
    # if img.shape[0] < yTarget:
    #     print('yt')
    #     img = cv2.copyMakeBorder(img, int((yTarget - yl) / 2), int((yTarget - yl) / 2), 0, 0, cv2.BORDER_REPLICATE)
    #
    # img = cv2.resize(img, (300, 200))
    #
    # print(img.shape)
    #
    # while True:
    #     ImgUtils.show('a', a, 0, 0)
    #     ImgUtils.show('b', b, 0, 300)
    #     ImgUtils.show('x', img, 0, 600)
    #     keyboard = cv2.waitKey(30)
    #     if keyboard == 'q' or keyboard == 27:
    #         break
