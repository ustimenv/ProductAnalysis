from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import cv2
from collections import OrderedDict
import numpy as np
from imgUtils import ImgUtils


class Tracker:
    def __init__(self):
        self.nextObjectId = 0                   # an ID number we will assign to the next object we detect(!=numObjects)
        self.numObjects = 0                     # actual number of objects that we have so far counted,
                                                # only incremented once we are sure we are seeing an actual object

        self.trackedCentroids = OrderedDict()   # objectID -> (centroidX, centroidY)
        self.durationTracked = OrderedDict()    # objectID -> consecutive frames tracked successfully
        self.missing = OrderedDict()            # objectID -> number of consecutive frames an object has been missing

        self.timeToDie = 30                     # consecutive frames till we stop tracking the object
        self.timeToLive = 3                     # consecutive frames till we start tracking the object

    # detected a previously untracked object
    def register(self, centroid):
        self.trackedCentroids[self.nextObjectId] = centroid
        self.missing[self.nextObjectId] = 0
        self.durationTracked[self.nextObjectId] = 1
        self.nextObjectId += 1

    # an object has been missing for too long
    def deregister(self, objectId):
        del self.trackedCentroids[objectId]
        del self.missing[objectId]
        del self.durationTracked[objectId]

    # detect rois either via cv2 or ML
    def track(self, rois):
        # 'special' case when no objects were detected, consider it to avoid a null pointer during the matching
        if rois is None or len(rois) == 0:
            for objectId in list(self.missing.keys()):
                self.missing[objectId] += 1
                if self.missing[objectId] > self.timeToDie:
                    self.deregister(objectId)
        else:
            # get centroids of the detected rois
            detectedCentroids = np.zeros((len(rois), 2), dtype="int")
            for i, roi in enumerate(rois):
                detectedCentroids[i] = ImgUtils.getCentroid(roi)

            # no centroids currently tracked, so start
            if len(self.trackedCentroids) == 0:
                for centroid in detectedCentroids:
                    self.register(centroid)

            # match detected centroids with tracked
            else:
                objectIds = list(self.trackedCentroids.keys())
                objectCentroids = list(self.trackedCentroids.values())

                D = cdist(np.array(objectCentroids), detectedCentroids)

                # in order to perform this matching we must (1) find the
                # smallest value in each row and then (2) sort the row
                # indexes based on their minimum values so that the row
                # with the smallest value is at the *front* of the index
                # list
                rows = D.min(axis=1).argsort()

                # next, we perform a similar process on the columns by
                # finding the smallest value in each column and then
                # sorting using the previously computed row index list
                cols = D.argmin(axis=1)[rows]

                usedRows = set()
                usedCols = set()

                # perform matching
                for row, col in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue

                    objectId = objectIds[row]
                    self.trackedCentroids[objectId] = detectedCentroids[col]
                    self.missing[objectId] = 0
                    self.durationTracked[objectId] += 1

                    usedRows.add(row)
                    usedCols.add(col)

                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                # in the event that the number of object centroids is
                # equal or greater than the number of input centroids
                # we need to check and see if some of these objects have
                # potentially disappeared
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

            for objectId in self.trackedCentroids:
                if self.durationTracked[objectId] == self.timeToLive:
                    self.numObjects += 1
        return self.trackedCentroids


