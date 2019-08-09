import numpy as np

from imgUtils import *


class PostCoolTrans:
    @staticmethod
    def transform(img):
        contrast = PostCoolTrans.extractChannel(img)[:, :, 2]
        return contrast

    @staticmethod
    def transformWithThresh(img):
        contrast = PostCoolTrans.extractChannel(img)[:, :, 2]
        contrast = PostCoolTrans.transformThresh(contrast)
        return contrast

    @staticmethod
    def extractChannel(img):
        channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # channel = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)   NOPE
        # channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)   NOPE
        # channel = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)   NOPE
        return channel


    @staticmethod
    def __transform(img):
        morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        contrast = PostCoolTrans.transformResize(img)
        contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
        contrast = cv2.GaussianBlur(contrast, (15, 15), 0)
        contrast = cv2.adaptiveThreshold(contrast[:, :, 2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        contrast = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, morphKernel, iterations=1)
        contrast = cv2.bitwise_not(contrast)
        return contrast

    @staticmethod
    def transformResize(img):
        return img[:, 530:830]
        # return np.copy(img[:, 530:830])

    @staticmethod
    def transformMorph(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        contrast = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel, iterations=1)
        return contrast

    @staticmethod
    def transformBlur(img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    @staticmethod
    def transformThresh(img):
        contrast = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return contrast

    @staticmethod
    def transformSobel(img, absolute=True, ksize=-1):
        return np.bitwise_or(PostCoolTrans.transformSobelX(img, absolute, ksize),
                             PostCoolTrans.transformSobelY(img, absolute, ksize))

    @staticmethod
    def transformSobelX(img, absolute=True, ksize = -1):
        if absolute:
            return np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)))
        else:
            return cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=ksize)

    @staticmethod
    def transformSobelY(img, absolute=True, ksize = -1):
        if absolute:
            return np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)))
        else:
            return cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=ksize)

    @staticmethod
    def transformLaplace(img, k=-1):
        return cv2.Laplacian(img, cv2.CV_64F, ksize=k)

