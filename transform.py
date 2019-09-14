
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

    def transform(self, img):
        # if self.transformationTarget == 'raw':
        #     contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        #     contrast = cv2.bitwise_not(contrast)
        #     contrast = self.transformBlur(contrast)
        #     contrast = cv2.threshold(src=contrast, type=cv2.THRESH_TOZERO, thresh=210, maxval=255)[1]

        if self.transformationTarget == 'raw':
            contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
            contrast = cv2.bitwise_not(contrast)
            contrast = cv2.boxFilter(src=contrast, ddepth=-1, ksize=(3, 17))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            contrast = cv2.morphologyEx(contrast, cv2.MORPH_DILATE, kernel, iterations=2)

            contrast = cv2.threshold(src=contrast, maxval=255, thresh=200, type=cv2.THRESH_BINARY)[1]

        elif self.transformationTarget == 'postbake':
            contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
            contrast = cv2.medianBlur(contrast, 7, 5)
            contrast = self.transformThresh(contrast)
        else:
            raise NotImplemented
        return contrast

    def transformResize(self, img):
        if self.transformationTarget == 'raw':
            return img[300:670, 250:760]
        elif self.transformationTarget == 'postbake':
            return img[250:, 530:830]

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


