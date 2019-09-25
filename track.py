from collections import OrderedDict

import numpy as np
from scipy.spatial.distance import cdist

from imgUtils import ImgUtils


# we track based only  on the y-ordinate: the conveyor is moving vertically so x-displacement is meaningless
# with PostCool, y decreases,
# with raw - increases


class Tracker:
    def __init__(self, lowerKillzone, upperKillzone, leftKillzone, rightKillzone, timeToDie, timeToLive,
                 roiTrackingMode):
        self.nextId = 0                         # an ID number we will assign to the next object we detect(!=numObjects)
        self.N = 0                              # actual number of objects that we have so far counted,
                                                # only incremented once we are sure we are seeing an actual object

        self.trackedCentroids = OrderedDict()   # objectID -> (centroidX, centroidY)
        self.missing = OrderedDict()            # objectID -> number of consecutive frames an object has been missing

        self.timeToDie = timeToDie              # consecutive frames till we stop tracking the object
        self.timeToLive = timeToLive            # consecutive frames till we start tracking the object
        self.roiTrackingMode = roiTrackingMode  # whether to track centroids or rois

        self.upperKillzone = upperKillzone                # border beyond which we do not accept any new detections
        self.lowerKillzone = lowerKillzone
        self.leftKillzone = leftKillzone
        self.rightKillzone = rightKillzone

        # if we are tracking rois, convert them to centroids
        if roiTrackingMode:
            self.roiConverter = self._roisToCentroidsHelper
        else:
            self.roiConverter = ImgUtils.returnUnchanged

    def register(self, centroid):
        bufferCondition = self.lowerKillzone < centroid[1] < self.upperKillzone \
                    and self.leftKillzone < centroid[0] < self.rightKillzone

        if bufferCondition:
            self.trackedCentroids[self.nextId] = centroid
            self.missing[self.nextId] = 0
            self.nextId += 1
            self.N += 1

    def deregister(self, objectId):
        del self.trackedCentroids[objectId]
        del self.missing[objectId]

    def _roisToCentroidsHelper(self, rois):
        centroids = np.zeros((len(rois), 2), dtype="int")
        for i, roi in enumerate(rois):
            centroids[i] = ImgUtils.getCentroid(roi)
        return centroids

    def track(self, detected):
        # 'special' case when no objects were detected, consider it to avoid a null pointer during the matching
        if detected is None or len(detected) == 0:
            for objectId in list(self.missing.keys()):
                self.missing[objectId] += 1
                if self.missing[objectId] > self.timeToDie:
                    self.deregister(objectId)
        else:
            detectedCentroids = self.roiConverter(detected)

            # no centroids currently tracked, so start
            if len(self.trackedCentroids) == 0:
                for centroid in detectedCentroids:
                    self.register(centroid)

            # match detected centroids with tracked
            else:
                objectIds = list(self.trackedCentroids.keys())
                objectCentroids = list(self.trackedCentroids.values())

                D = cdist(np.array(objectCentroids), detectedCentroids)
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
                    if abs(self.trackedCentroids[objectId][1]-detectedCentroids[col][1]) < 100:
                        self.trackedCentroids[objectId] = detectedCentroids[col]
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
                        self.register(detectedCentroids[col])

        return self.trackedCentroids
