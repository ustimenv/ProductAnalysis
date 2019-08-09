from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import cv2
from collections import OrderedDict
import numpy as np
from imgUtils import ImgUtils

# we track based only  on the y-ordinate: the conveyor is moving vertically so x-displacement is meaningless
# with PostCool, y decreases,
# with raw - increases
class __Tracker:
    def __init__(self, killzone=100, timeToDie=0, timeToLive=1):
        self.tracked = OrderedDict()        # objectId -> y-ordinate
        self.durationTracked = OrderedDict()
        self.timeToLive = timeToLive        # consecutive frames object is present till we start tracking
        self.N = 0                          # total number of objects we have counted so far
        self.nextId = 0

        self.missing = OrderedDict()
        self.timeToDie = timeToDie          # consecutive frames object is missing till we stop tracking
        self.killzone = killzone            # y coordinate at which we can reliably 'untrack' objects,
                                            # do not start tracking there

    def register(self, y):                  # start tracking but don't assume this is an object just yet
        if not self._isInKillzone(y):
            self.tracked[self.nextId] = y
            self.missing[self.nextId] = 0
            self.durationTracked[self.nextId] = 1
            self.nextId += 1

    def deregister(self, objectId):
        del self.tracked[objectId]
        del self.missing[objectId]
        del self.durationTracked[objectId]

    def track(self, detectedRois):
        if detectedRois is not None and len(detectedRois) > 0:
            detected = [ImgUtils.getCentroid(roi)[1] for roi in detectedRois]
            if len(self.tracked) < 1:               # not tracking anything atm
                for d in detected:                  # so start tracking
                    self.register(d)

            # the juicy stuff, match tracked with the detected
            else:
                unusedDetects = set(detected)
                unusedTracked = set(self.tracked.values())
                detected.sort()
                tracked = list(self.tracked.values())
                tracked.sort()
                for d, t in zip(detected, tracked):
                    if Tracker._isValidMatchingWide(d, t):          # successful matching
                        for objectId in list(self.tracked):
                            if self.tracked[objectId] == t:
                                self.tracked[objectId] = d
                                self.durationTracked[objectId] += 1
                                self.missing[objectId] = 0
                                break

                        unusedDetects.remove(d)
                        unusedTracked.remove(t)

                # previously untracked object
                for d in unusedDetects:
                    if d < self.killzone:
                        continue
                    self.register(d)

                # we appear to have lost an object
                for t in unusedTracked:
                    for objectId in list(self.tracked):
                        if self.tracked[objectId] == t:
                            self.missing[objectId] += 1
                            self.durationTracked[objectId] = 0

                        if self.missing[objectId] > self.timeToDie:
                            self.deregister(objectId)

        # no observations
        else:
            for objectId in list(self.missing):
                self.missing[objectId] += 1
                if self.missing[objectId] > self.timeToDie:
                    self.deregister(objectId)

            if self.tracked is None or len(self.tracked) < 1:       # nothing at all, the belt must be empty
                pass
                # print('Empty')

        # for objectId in list(self.tracked):
        #     if self._isInKillzone(self.tracked[objectId]):
                # print('Deleting')
                # del self.tracked[objectId]
        return self.tracked

    def _isInKillzone(self, yOrdinate):
        return yOrdinate < self.killzone

    @staticmethod
    def _isValidMatching(detectedY, trackedY):
        # the new detection must be 'above' the tracked and within a sensible range
        return  abs(detectedY-trackedY) < 50

    @staticmethod
    def _isValidMatchingWide(detectedY, trackedY):
        # the new detection must be 'above' the tracked and within a sensible range
        return detectedY - trackedY < 5 or trackedY > detectedY


class Tracker:
    def __init__(self, movementDirection, killzone, lowerKillzone,  timeToDie, timeToLive):
        self.nextId = 0                         # an ID number we will assign to the next object we detect(!=numObjects)
        self.N = 0                              # actual number of objects that we have so far counted,
                                                # only incremented once we are sure we are seeing an actual object

        self.trackedCentroids = OrderedDict()   # objectID -> (centroidX, centroidY)
        self.missing = OrderedDict()            # objectID -> number of consecutive frames an object has been missing

        self.timeToDie = timeToDie              # consecutive frames till we stop tracking the object
        self.timeToLive = timeToLive            # consecutive frames till we start tracking the object
        self.killzone = killzone                # border beyond which we do not accept any new detections
        self.lowerKillzone = lowerKillzone

        self.movementDirection = movementDirection  # +1 if products move up, -1 if down

    def register(self, centroid):
        bufferCondition = self.lowerKillzone < centroid[1] < self.killzone

        if bufferCondition:
            self.trackedCentroids[self.nextId] = centroid
            self.missing[self.nextId] = 0
            self.nextId += 1
            self.N += 1

    def deregister(self, objectId):
        del self.trackedCentroids[objectId]
        del self.missing[objectId]

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
                # indices based on their minimum values so that the row
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
                    #####
                    if abs(self.trackedCentroids[objectId][1]-detectedCentroids[col][1]) < 100:
                        self.trackedCentroids[objectId] = detectedCentroids[col]
                        self.missing[objectId] = 0
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

        return self.trackedCentroids
