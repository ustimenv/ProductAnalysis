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
            for fullPath in glob.glob('/home/vlad/Work/1/MLworkDir/%s/*/**.png' % lstName, recursive=True):
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
    # def padAll(path='/home/vlad/Work/1/MLworkDir/', xThresh=140, yThresh=140):
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

                        imgPadded, bbox = SystemUtils.pad(cv2.imread(fullPath))
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



if __name__ == "__main__":
    # SystemUtils.initLST('/home/vlad/Work/1/MLworkDir', 'Train')
    # SystemUtils.initLST('/home/vlad/Work/1/MLworkDir', 'Test')
    PerClass = {      '0' : {'numberToCopy' : 6,
                             'widthThresh' : 140,
                             'heightThresh' : 140,

                             },

                      '1' : {'numberToCopy' : 18,
                             'widthThresh': 140,
                             'heightThresh': 140,

                             },


                }

    SystemUtils.fullSetup(srcDir='/home/vlad/Work/1/MlMasterDir',
                          dstDir='/home/vlad/Work/1/MlWorkDir',
                          postfix='Miny', perClass=PerClass)

    pass
    # SystemUtils.copySomeV2("/home/vlad/Work/1/raw/UnitSamples",
    #                         "/home/vlad/Work/1/Train/0",
    #                         "/home/vlad/Work/1/Test/0",
    #                         140, 140, 10)
    #
    # SystemUtils.copySomeV2("/home/vlad/Work/1/postbake/UnitSamples",
    #                         "/home/vlad/Work/1/Train/1",
    #                         "/home/vlad/Work/1/Test/1",
    #                         140, 140, 10)