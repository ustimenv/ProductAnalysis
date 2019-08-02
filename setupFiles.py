import glob
from os import path
from subprocess import call

import cv2


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
        from numpy import hstack
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
    def initDir(topDirName, scaleFactor=5):
        for i in range(1, 10):
            call('mkdir %s/%s' %(topDirName, str(i)), shell=True)
        SystemUtils.copySome(topDirName, scaleFactor)
        print('File moved')
        SystemUtils.initLST(topDirName)
        print('LST complete')
        SystemUtils.initRec(topDirName)
        print('REC complete')

    @staticmethod
    def initLST(topDirName):  # (xmin, ymin, xmax, ymax)
        bbTargets = {'1': (70, 90, 250, 170),  # sochen yagodny
                     '2': (70, 90, 250, 170),  # sochen tvorog
                     '3': (70, 90, 250, 190),  # strudel vishnya+
                     '4': (70, 90, 250, 190),  # strudel malina+
                     '5': (50, 100, 255, 200),  # sochen yagodnya+
                     '6': (90, 100, 230, 200),  # vatrushki +
                     '7': (40, 60, 280, 230),  # novomosk +
                     '8': (50, 100, 260, 190),  # lakomka +
                     '9': (70, 100, 240, 210)}  # korj+

        bbTargomente = {'1': [122322.0, 190281.33333333334, 547725.5, 392778.3333333333],
                        '2': [108154.0, 180342.0, 565860.5, 414921.0],
                        '3': [14347.0, 176526.0, 644355.0, 536023.0],
                        '4': [14477.0, 181667.0, 657200.0, 532675.5],
                        '5': [141348.0, 205541.5, 572919.5, 405641.0],
                        '6': [11836.5, 175868.0, 640601.0, 503967.0],
                        '7': [13920.166666666668, 134374.5, 611991.6666666666, 613755.0],
                        '9': [34008.5, 170826.0, 597611.0, 462383.0],
                        '8': [125267.0, 202924.0, 573527.0, 424928.0]}

        numElems = {'1': 2009, '2': 2054, '3': 2149, '4': 2193, '5': 2120, '6': 2150, '7': 2154, '8': 2147, '9': 2061}

        xTargets = {'1': (65, 105, 260, 190),
                    '2': (65, 100, 260, 190),
                    '3': (70, 90, 250, 180),
                    '4': (70, 90, 250, 180),
                    '5': (65, 110, 260, 180),
                    '6': (96, 100, 230, 200),
                    '7': (60, 80, 270, 230),
                    '8': (65, 110, 260, 180),
                    '9': (95, 90, 250, 200)
                    }

        import numpy as np

        with open('%s/annoTrainX.lst' % (topDirName), 'w') as fw:
            counter = 0

            for fullPath in glob.glob('%s/*/*.png' % (topDirName), recursive=True):
                # print(fullPath)
                dirName = fullPath.split('/')[1]

                counter += 1

                img = cv2.imread(fullPath)
                fullPath = fullPath[len(topDirName)+1:]
                # print(fullPath)
                if img is None:
                    continue

                line = SystemUtils.writeLSTLine(fullPath,
                                                img.shape,
                                                np.array([xTargets[dirName]], ),
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
