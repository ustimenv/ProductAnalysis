import glob
import os
import cv2
import sys
import numpy as np
from PIL import Image
from PIL.ImagePath import Path

from imgUtils import *
from gluoncv.data import batchify


class RawDetectionTrans:
    @staticmethod
    def transform(img):
        contrast = img
        contrast = RawDetectionTrans.extractChannel(contrast)[:, :, 0]
        contrast = RawDetectionTrans.transformThresh(contrast)
        return contrast

    @staticmethod
    def _transform(img):
        contrast = RawDetectionTrans.transformResize(img)
        contrast = RawDetectionTrans.extractChannel(contrast)[:, :, 1]
        contrast = RawDetectionTrans.transformMorph(contrast)
        contrast = RawDetectionTrans.transformThresh(contrast)
        return contrast

    @staticmethod
    def extractChannel(img):
        # channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        ##channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        ##channel = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ##channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        # channel = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return channel



    @staticmethod
    def transformResize(img):
        return np.copy(img[:, 500:760])

    @staticmethod
    def transformMorph(img):
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        img = cv2.morphologyEx(img, cv2.MORPH_HITMISS, morphKernel, iterations=1)
        return img

    @staticmethod
    def transformBlur(img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    @staticmethod
    def transformThresh(img):
        contrast = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return contrast

    @staticmethod
    def transformSobel(img, absolute=True):
        return np.bitwise_or(RawDetectionTrans.transformSobelX(img, absolute), RawDetectionTrans.transformSobelY(img, absolute))

    @staticmethod
    def transformSobelX(img, absolute=True):
        if absolute:
            return np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)))
        else:
            return cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

    @staticmethod
    def transformSobelY(img, absolute=True):
        if absolute:
            return np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)))
        else:
            return cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=5)

    @staticmethod
    def transformLaplace(img):
        return cv2.Laplacian(img, cv2.CV_64F)

