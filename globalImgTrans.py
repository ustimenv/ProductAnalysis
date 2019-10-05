import glob
import os

from PIL import Image
from PIL.ImagePath import Path

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
            if averageX > padDims:
                xMax = averageX
            if averageY > padDims:
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

    @staticmethod
    def padGlobV2(padDims=300):
        for filename in glob.glob('/home/vlad/Work/1/Train/1/**', recursive=False):

            img = cv2.imread(filename)

            imgName = filename.split('/')[-1]
            h, w, _ = img.shape
            gmi = cv2.copyMakeBorder(img, top=300-h, bottom=300-h,
                                     left=300-w, right=300-w,
                                     borderType=cv2.BORDER_CONSTANT)

            while True:
                ImgUtils.show('Orig', img, 0, 0)
                ImgUtils.show('Then', gmi, 0, 400)
                key = cv2.waitKey(13)
                if key == ord('q'):
                    return
                elif key == ord('v'):
                    break

if __name__ == "__main__":
    GlobImgTrans.padGlobV2()