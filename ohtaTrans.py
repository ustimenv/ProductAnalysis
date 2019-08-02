import glob
import os
import cv2
import sys
import numpy as np
from PIL import Image
from PIL.ImagePath import Path

from imgUtils import *
from gluoncv.data import batchify


class StandardDetectionTrans:
    # @staticmethod
    # def prepareMono(img):
    #     morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    #     contrast = ImageTransforms.prepareResized(img)
    #     contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
    #     contrast = cv2.GaussianBlur(contrast, (15, 15), 0)
    #     contrast = cv2.adaptiveThreshold(contrast[:, :, 2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    #     contrast = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, morphKernel, iterations=1)
    #     contrast = cv2.bitwise_not(contrast)
    #     return contrast
    @staticmethod
    def prepareMono(img):
        imgs = []
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        contrast = StandardDetectionTrans.prepareResized(img)
        #HSV, HLS
        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2HLS)
        imgs.append(contrast)
        # contrast = cv2.GaussianBlur(contrast, (15, 15), 0)
        # contrast = cv2.adaptiveThreshold(contrast[:, :, 2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        # contrast = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, morphKernel, iterations=1)
        # contrast = cv2.bitwise_not(contrast)
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
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        contrast = cv2.GaussianBlur(contrast, (15, 15), 0)
        c2 = cv2.adaptiveThreshold(contrast[:, :, 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        c2 = cv2.morphologyEx(c2, cv2.MORPH_OPEN, morphKernel, iterations=1)
        contrast = cv2.bitwise_not(c2)
        return contrast

    @staticmethod
    def prepareContrastPHZ(img):
        contrast = cv2.threshold(img[:,:, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        print(len(contrast))
        return contrast


    # @staticmethod
    # def PHZMono(img):
    #     img = img[:, 550:-450]
    #     morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    #     contrast = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     contrast = cv2.GaussianBlur(contrast, (5, 5), 0)
    #     contrast = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
    #     contrast = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, morphKernel, iterations=1)
    #     contrast = ImageTransforms.prepareContrastPHZ(contrast)
    #     contrast = cv2.bitwise_not(contrast)
    #     return contrast

    @staticmethod
    def PHZMono(img):
        img = img[:, 500:-200]
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
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
        newImg = StandardDetectionTrans.prepareResized(img)
        newImg = StandardDetectionTrans.sobelTransform(newImg)
        newImg = StandardDetectionTrans.prepareBlur(newImg)
        newImg = StandardDetectionTrans.prepareContrast(newImg)
        newImg = StandardDetectionTrans.prepareMorph(newImg)
        return newImg

    @staticmethod
    def sobelTransform(img, absolute=True):
        return np.bitwise_or(StandardDetectionTrans.sobelTransformX(img, absolute), StandardDetectionTrans.sobelTransformY(img, absolute))


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

