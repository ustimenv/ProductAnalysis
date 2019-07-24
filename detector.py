import cv2
import sys
import numpy as np

from imageTrans import *


class Detector:
    def __init__(self):
        pass

    def detect(self, contrast):
        contours, h = cv2.findContours(contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        currentRois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) == 0 or w < 60 or h < 30 or h > w or w>400 or h>200:
                continue
            #coordinates of the current bounding box
            x1 = x #- 40
            x2 = x1 + w #+ 60
            y1 = y #- 20
            y2 = y1 + h# +50
            currentRois.append((x1, y1, x2, y2))
        return currentRois


