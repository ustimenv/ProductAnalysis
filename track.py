from collections import OrderedDict

import numpy as np
from scipy.spatial.distance import cdist

from utils.imgUtils import ImgUtils


# we track based only  on the y-ordinate: the conveyor is moving vertically so x-displacement is meaningless
# with PostCool, y decreases,
# with raw - increases


class Tracker:
    def __init__(self, lowerBound, upperBound, leftBound, rightBound, timeToDie, timeToLive):
        self.nextId = 0                         # an ID number we will assign to the next object we detect(!=numObjects)
        self.N = 0                              # actual number of objects that we have so far counted,
                                                # only incremented once we are sure we are seeing an actual object

        self.trackedRois = OrderedDict()        # objectID -> (centroidX, centroidY)
        self.missing = OrderedDict()            # objectID -> number of consecutive frames an object has been missing

        self.timeToDie = timeToDie              # consecutive frames till we stop tracking the object
        self.timeToLive = timeToLive            # consecutive frames till we start tracking the object

        self.upperBound = upperBound      # border beyond which we do not accept any new detections
        self.lowerBound = lowerBound
        self.leftBound = leftBound
        self.rightBound = rightBound

    def register(self, roi):
        bufferCondition = self.lowerBound < (roi[1]+roi[3])/2 < self.upperBound \
                          and self.leftBound < (roi[0]+roi[2])/2 < self.rightBound

        if bufferCondition:
            self.trackedRois[self.nextId] = roi
            self.missing[self.nextId] = 0
            self.nextId += 1
            self.N += 1
            return roi

    def deregister(self, objectId):
        del self.trackedRois[objectId]
        del self.missing[objectId]

    # def _roisToCentroidsHelper(self, rois):
    #     centroids = np.zeros((len(rois), 2), dtype="int")
    #     for i, roi in enumerate(rois):
    #         centroids[i] = ImgUtils.getCentroid(roi)
    #     return centroids

    def track(self, detected):
        newDetections = []
        # 'special' case when no objects were detected, consider it to avoid a null pointer during the matching
        if detected is None or len(detected) == 0:
            for objectId in list(self.missing.keys()):
                self.missing[objectId] += 1
                if self.missing[objectId] > self.timeToDie:
                    self.deregister(objectId)
        else:
            # detections = self.roiConverter(detected)

            # no centroids currently tracked, so start
            if len(self.trackedRois) == 0:
                for det in detected:
                    det = self.register(det)
                    if det is not None:
                        newDetections.append(det)

            # match detected centroids with tracked
            else:
                objectIds = list(self.trackedRois.keys())
                objectRois = list(self.trackedRois.values())

                D = cdist(np.array(objectRois), detected)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()

                # perform matching
                for row, col in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue
                    objectId = objectIds[row]
                    #####
                    if abs(
                            ImgUtils.findRoiCentroid(self.trackedRois[objectId])[1] -
                            ImgUtils.findRoiCentroid(detected[col])[1]) < 100:

                        self.trackedRois[objectId] = detected[col]
                        self.missing[objectId] = 0
                        usedRows.add(row)
                        usedCols.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        objectId = objectIds[row]
                        self.missing[objectId] += 1
                        # check to see if the number of consecutive
                        # frames the object has been marked "disappeared"
                        # for warrants deregistering the object
                        if self.missing[objectId] > self.timeToDie:
                            self.deregister(objectId)

                # otherwise, if the number of input centroids is greater
                # than the number of existing object centroids we need to
                # register each new input centroid as a trackable object
                else:
                    for col in unusedCols:
                        det = self.register(detected[col])
                        if det is not None:
                            newDetections.append(det)

        return self.trackedRois, newDetections
