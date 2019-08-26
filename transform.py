
import glob
import os
import cv2
import sys
import numpy as np
from PIL import Image
from PIL.ImagePath import Path

from imgUtils import *
from gluoncv.data import batchify


class Transformer:
    def __init__(self, transformationTarget):
        self.transformationTarget = transformationTarget
        self.rawKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

    def transform(self, img):
        if self.transformationTarget == 'raw':
            contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
            contrast = cv2.bitwise_not(contrast)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

            contrast = cv2.threshold(src=contrast, type=cv2.THRESH_TOZERO, thresh=210, maxval=255)[1]
            # contrast = cv2.morphologyEx(self.transformBlur(contrast), cv2.MORPH_ERODE, kernel, iterations=1)


            # contrast = cv2.inRange(contrast, 165, 255)

            # contrast = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 1]
            # contrast = cv2.inRange(contrast, 80, 112)
            # contrast = cv2.bitwise_not(contrast)
            #
            # contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
            # contrast3 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 1]
            # contrast2 = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 2]
            # contrast = self.transformMorph(contrast)
            # contrast  = cv2.adaptiveThreshold(maxValue=40, src=contrast, blockSize=12, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C);
            # contrast = cv2.bitwise_not(contrast)
            # contrast = self.transformThresh(contrast)
            # contrast = cv2.inRange(contrast, 0, 160)

        elif self.transformationTarget == 'cool':
            contrast = self.extractChannel(img)[:, :, 2]
            contrast = self.transformThresh(contrast)
        else:
            raise NotImplemented
        return contrast

    def extractChannel(self, img):
        channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return channel

    def transformResize(self, img):
        if self.transformationTarget == 'raw':
            # return img[420:620, :]
            return img[440:620, 240:-640]
        elif self.transformationTarget == 'cool':
            return img[:, 530:830]

    def transformMorph(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        contrast = cv2.morphologyEx(self.transformBlur(img), cv2.MORPH_CLOSE, kernel, iterations=1)
        return contrast

    def transformBlur(self, img):
        return cv2.GaussianBlur(img, (45, 5), 0)

    def testBlur(self, img, x, y, z):
        return cv2.GaussianBlur(img, (x, y), z)

    def transformThresh(self, img):
        contrast = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return contrast



    def testThresh2(self, img, a, b):
        # contrast = cv2.threshold(contrast, a, b, cv2.THRESH_TOZERO)[1]
        contrast = cv2.threshold(img, a, b, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return contrast

