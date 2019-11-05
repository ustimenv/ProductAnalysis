import glob
import sys
from os import path
from subprocess import call
import numpy as np
from PIL import Image
from numpy import hstack
import traceback

import cv2
from numpy.distutils.system_info import x11_info

from detectorFactory import DetectorFactory
from imgUtils import ImgUtils


class SystemUtils:
    @staticmethod
    def initRecFiles(imageTypes=('Data',), subFolderTypes=("Train", "Test"), recFileNames=("train", "test")):
        createListCommand = 'python3.5 ~/PythonProjects/OpenCV/im2rec.py ~/PythonProjects/OpenCV/ProductTracker/%s/%s/%s \
                             ~/PythonProjects/OpenCV/ProductTracker/%s/%s/ --recursive --list --num-thread 8'

        createRecCommand = 'python3.5 ~/PythonProjects/OpenCV/im2rec.py ~/PythonProjects/OpenCV/ProductTracker/%s/%s/%s  \
                           ~/PythonProjects/OpenCV/ProductTracker/%s/%s/ --recursive --pass-through --pack-label --num-thread 8'

        commands = (createListCommand, createRecCommand)

        for x in commands:
            for y in imageTypes:
                for (z1, z2) in zip(subFolderTypes, recFileNames):
                    call(x % (y, z1, z2, y, z1), shell=True)


    @staticmethod
    def createDir(dirPath):
        path.os.makedirs(dirPath, exist_ok=False)

    @staticmethod
    def writeLSTLine(img_path, im_shape, boxes, ids, idx, normaliseBboxes=True):
        h, w, c = im_shape
        # for header, we use minimal length 2, plus width and height
        # with A: 4, B: 5, C: width, D: height
        A = 4
        B = 5
        C = w
        D = h
        # concat id and bboxes
        labels = hstack((ids.reshape(-1, 1), boxes)).astype('float')
        if normaliseBboxes:
            labels[:, (1, 3)] /= float(w)
            labels[:, (2, 4)] /= float(h)
        # flatten
        labels = labels.flatten().tolist()
        str_idx = [str(idx)]
        str_header = [str(x) for x in [A, B, C, D]]
        str_labels = [str(x) for x in labels]
        str_path = [img_path]
        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line



    @staticmethod
    def getDirElemsNums():
        for outerDirs in glob.glob('Data/Train/*', recursive=False):
            elemsByDir= {outerDirs.split('/')[2] : 0}
            for fullPath in glob.glob('Data/Train/**/*.png', recursive=True):
                category = fullPath.split('/')[2]
                if category in elemsByDir.keys():
                        elemsByDir[category]+=1
                else:   #failsafe
                    elemsByDir[category]=1

    @staticmethod
    def initLST(topDirName, lstName):             # (xmin, ymin, xmax, ymax)
        with open('%s/%sx.lst' % (topDirName, lstName), 'w') as fw:
            counter = 0
            for fullPath in glob.glob('/beta/Work/1/MLworkDir/%s/*/**.png' % lstName, recursive=True):
                counter += 1
                dirName = fullPath.split('/')[-2]

                gmi, bbox = SystemUtils.pad(cv2.imread(fullPath))
                if gmi is None:
                    continue

                cv2.imwrite(fullPath, gmi)

                line = SystemUtils.writeLSTLine(fullPath,
                                                gmi.shape,
                                                np.array([bbox], ),
                                                np.array([int(dirName)]),
                                                counter, normaliseBboxes=True)
                print(line)
                fw.write(line)

    @staticmethod
    def initRec(topDirName, lstFileName=""):
        createRecCommand = 'python3.5 ~/PythonProjects/OpenCV/im2rec.py ' \
                           ' ~/PythonProjects/OpenCV/ProductTracker/%s/annoTrainX '\
                            '~/PythonProjects/OpenCV/ProductTracker/%s/ ' \
                           '--recursive --pass-through --pack-label --num-thread 8'

        call(createRecCommand %(topDirName, topDirName), shell=True)

    @staticmethod
    def copySome(topDirName, scaleFactor=10):
        command = 'cp ~/PythonProjects/OpenCV/ProductTracker/Data/Train/%s/%s ' \
                    ' ~/PythonProjects/OpenCV/ProductTracker/%s/%s/'

        for outerDirs in glob.glob('Data/Train/*', recursive=False):
            elemsByDir= {outerDirs.split('/')[2] : 0}

            for fullPath in glob.glob('Data/Train/**/*.png', recursive=True):
                print(fullPath)
                category = fullPath.split('/')[2]
                filename = fullPath.split('/')[3]
                numComponent = filename[:-7]
                # print(category, filename, numComponent)
                # print(numComponent)
                if int(numComponent) % scaleFactor == 0:
                    call(command %(category, filename, topDirName, category), shell=True)

    @staticmethod
    def copySomeV2(srcDirName, trainDirName, testDirName, xThresh, yThresh, trainToTestRatio):
        counter = 0
        command = "cp %s %s/%s"

        for filename in glob.glob(srcDirName+"/**.png"):
            img = cv2.imread(filename=filename)
            if img is None:
                continue

            if img.shape[0] < yThresh or img.shape[1] < xThresh:
                continue

            counter += 1
            imgName = filename.split('/')[-1]

            if counter % trainToTestRatio == 0:
                call(command % (filename, testDirName, imgName), shell=True)
            else:
                call(command % (filename, trainDirName, imgName), shell=True)

    @staticmethod
    def pad(img, newDim=300):
        h, w, _ = img.shape

        try:
            gmi = cv2.copyMakeBorder(src=img, top=0, bottom=newDim-h, left=0, right=newDim-w,
                                     borderType=cv2.BORDER_CONSTANT)
        except:
            # gmi = cv2.resize(img, dsize=(newDim, newDim))
            gmi = None
            print('error resizing')
        return gmi, (0, 0, w, h)

    # @staticmethod
    # def padAll(path='/beta/Work/1/MLworkDir/', xThresh=140, yThresh=140):
    #     path = path+'*/**/*.png'
    #     nDeleted=0
    #     for imgPath in glob.glob(path, recursive=False):
    #         img = cv2.imread(imgPath)
    #         h, w, _ = img.shape
    #         if w < xThresh or h < yThresh:
    #             call('rm %s' % imgPath, shell=True)
    #             nDeleted += 1
    #             continue
    #
    #         img = SystemUtils.pad(img, 300)[0]
    #         if img is None:
    #             call('rm %s' % imgPath, shell=True)
    #             nDeleted += 1
    #         else:
    #             cv2.imwrite(filename=imgPath, img=img)
    #
    #     print('Deleted Total: ', nDeleted)

    @staticmethod
    def fullSetup(srcDir, dstDir, postfix, perClass):
        if True:
            try:
                trainDirName = dstDir + '/Train' + postfix
                testDirName = dstDir + '/Test' + postfix

                call('mkdir %s %s' % (trainDirName, testDirName), shell=True)
                for category in perClass.keys():
                    call('mkdir %s %s' % (trainDirName + '/' + category,
                                          testDirName + '/' + category), shell=True)

            except Exception as e:
                traceback.print_exc()

            print('Directories initialised successfully')

        if True:
            copyCommand = "cp %s %s"
            pendulum = 0
            counter = 0
            for _srcFilename in glob.glob(srcDir + "/**/*.png", recursive=True):
                srcFilename = _srcFilename.split('/')
                imgClass = srcFilename[-2]

                dstFilename = dstDir+'/'+srcFilename[-3]+postfix+'/'+srcFilename[-2]+'/'+srcFilename[-1]

                w, h = Image.open(_srcFilename).size
                try:
                    if w < perClass[imgClass]['widthThresh'] or h < perClass[imgClass]['heightThresh']:
                        continue
                except Exception as e:
                    traceback.print_exc()
                    continue

                pendulum += 1
                if pendulum % perClass[imgClass]['numberToCopy'] == 0:
                    counter += 1
                    if counter % 20 == 0:
                        call(copyCommand % (_srcFilename, dstFilename), shell=True)
                    else:
                        call(copyCommand % (_srcFilename, dstFilename), shell=True)

            print('Directories filled successfully')

        if True:
            for target in ('Test', 'Train'):
                target += postfix
                with open('%s/%s.lst' % (dstDir, target), 'w') as fw:
                    index = 0
                    for fullPath in glob.glob(dstDir+'/%s/**/*.png' % target, recursive=True):
                        imgClass = fullPath.split('/')[-2]

                        #imgPadded, bbox = SystemUtils.pad(cv2.imread(fullPath), newDim=512)
                        # imgPadded = cv2.resize(cv2.imread(fullPath), (300, 300))

                        if imgPadded is None:
                            continue

                        cv2.imwrite(fullPath, imgPadded)
                        index += 1
                        line = SystemUtils.writeLSTLine(fullPath,
                                                        imgPadded.shape,
                                                        np.array([bbox], ),
                                                        np.array([int(imgClass)]),
                                                        index, normaliseBboxes=True)
                        fw.write(line)
                print('Images resized successfully')
                print('LST files generated')

                createRecCommand = 'python3.5 ~/Work/im2rec.py ' \
                                   '%s/%s.lst ' \
                                   '%s '\
                                   '--recursive --pass-through --pack-label --num-thread 8'

                call(createRecCommand % (dstDir, target, '/'), shell=True)
                print('REC completed')


class Setup:
    def __init__(self, srcDir, dstDir, postfix):
        self.srcDir = srcDir
        self.postfix = postfix
        self.dstDir = dstDir

        self.PerClass = {   '0': {'name': 'raw',
                                  'line' : 1,
                               'numberToCopy': 1,

                                 },
                            '1': {'name': 'postbake',
                                  'line': 1,
                                  'numberToCopy': 3,
                                  },
                            '2': {'name': 'brick',
                                  'line': 3,
                                  'numberToCopy': 3,
                                  },
                    }

        self.extactors = dict()
        for cat in self.PerClass.keys():
            self.extactors[cat] = DetectorFactory.getDetector(self.PerClass[cat]['line'], self.PerClass[cat]['name'])

    def extractAndCopy(self, srcDirPath, extractor=0):
        counter=0

        for i in range(10000):
            imgPath = srcDirPath + str(i)+'.png'
            # print(imgPath)
            img = cv2.imread(imgPath)
            if img is None:
                continue
            dets = self.extactors[str(extractor)].detectDebug(img)
            # print(len(dets))
            for det in dets:
                ImgUtils.drawRect(det, img)
                try:
                    counter += 1
                except:
                    print("out of bounds")
            if True:
                ImgUtils.show('Img', img, 0, 0)
                key = cv2.waitKey(30)
                if key == ord('v'):
                    break
                elif key == ord('q'):
                    return
        return

        counter=0
        filepathPrefix = "/beta/Work/2/Train/postbake/"
        for imgName in glob.glob("/beta/Work/2/postbake/*.png", recursive=True):
            img = cv2.imread(imgName)
            dets = self.D.detectDebug(img)
            for (cX, cY), rad in dets:
                if 600 < cX < 760 and 160 < cY < 560:
                    cv2.imwrite(filepathPrefix+str(counter)+".png",
                                ImgUtils.sampleAround(img, cX, cY, 300, 300))
                    counter += 1

            while True:
                ImgUtils.show('Img', img, 0, 0)
                # if X is not None:
                #     ImgUtils.show('X', X, 800, 900)
                key = cv2.waitKey(30)
                if key == 27 or key == ord('v'):
                    break
                elif key == ord('q'):
                    return

    def initDirs(self):
        try:
            trainDirName = self.dstDir + '/Train' + self.postfix
            testDirName = self.dstDir + '/Test' + self.postfix

            call('mkdir %s %s' % (trainDirName, testDirName), shell=True)
            for category in self.perClass.keys():
                call('mkdir %s %s' % (trainDirName + '/' + category,
                                      testDirName + '/' + category), shell=True)

        except Exception as e:
            traceback.print_exc()
        print('Directories initialised successfully')

    def copy(self):
        copyCommand = "cp %s %s"
        pendulum = 0
        counter = 0
        for _srcFilename in glob.glob(self.srcDir + "/**/*.png", recursive=True):
            srcFilename = _srcFilename.split('/')
            imgClass = srcFilename[-2]

            dstFilename = self.dstDir + '/' + srcFilename[-3] + self.postfix + '/' + srcFilename[-2] + '/' + srcFilename[-1]

            w, h = Image.open(_srcFilename).size
            try:
                if w < self.perClass[imgClass]['widthThresh'] or h < self.perClass[imgClass]['heightThresh']:
                    continue
            except Exception as e:
                traceback.print_exc()
                continue

            pendulum += 1
            if pendulum % self.perClass[imgClass]['numberToCopy'] == 0:
                counter += 1
                if counter % 20 == 0:
                    call(copyCommand % (_srcFilename, dstFilename), shell=True)
                else:
                    call(copyCommand % (_srcFilename, dstFilename), shell=True)

        print('Directories filled successfully')

    def initLst(self):
        for target in ('Train', ):
            target += self.postfix
            with open('%s/%s.lst' % (self.dstDir, target), 'w') as fw:
                index = 0
                for fullPath in glob.glob(self.dstDir+'/**/*.png', recursive=True):
                    imgNumClass = fullPath.split('/')[-2]

                    img = cv2.imread(fullPath)
                    dets = self.extactors[imgNumClass].detectDebug(img)
                    if dets is None or len(dets) != 1:
                        continue

                    index += 1
                    line = SystemUtils.writeLSTLine(fullPath,
                                                    img.shape,
                                                    np.array([dets[0]], ),
                                                    np.array([int(imgNumClass)]),
                                                    index, normaliseBboxes=False)
                    print(line)
                    fw.write(line)
            print('LST file(s) generated')

            createRecCommand = 'python3.5 /beta/Work/im2rec.py ' \
                               '%s/%s.lst ' \
                               '%s ' \
                               '--recursive --pass-through --pack-label --num-thread 8'

            call(createRecCommand % (self.dstDir, target, '/'), shell=True)
            print('REC completed')


if __name__ == "__main__":
    C = Setup(srcDir='/beta/Work/2/Train',
              dstDir='/beta/Work/2/Train',
              postfix='Z')
    C.extractAndCopy(srcDirPath='/beta/Work/2/brick/', extractor=2)

    # C.initLst()

    # SystemUtils.fullSetup(srcDir='/beta/Work/2/MlMasterDir',
    #                       dstDir='/beta/Work/2/MlWorkDir',
    #                       postfix='Full', perClass=PerClass)
