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
        if self.transformationTarget == 'raw':
            contrast = self.extractChannel(img)[:, :, 0]
            contrast = self.transformMorph(contrast)
        elif self.transformationTarget == 'cool':
            contrast = self.extractChannel(img)[:, :, 2]
        else:
            raise NotImplemented

        contrast = self.transformThresh(contrast)
        return contrast

    def extractChannel(self, img):
        channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        return channel

    def transformResize(self, img):
        if self.transformationTarget == 'raw':
            return img[:, 500:760]
        elif self.transformationTarget == 'cool':
            return img[:, 530:830]

    def transformMorph(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        contrast = cv2.morphologyEx(self.transformBlur(img), cv2.MORPH_DILATE, kernel, iterations=1)
        return contrast

    def transformBlur(self, img):
        return cv2.GaussianBlur(img, (45, 5), 0)
     
    def transformThresh(self, img):
        contrast = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return contrast

