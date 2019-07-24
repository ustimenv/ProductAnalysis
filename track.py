from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from collections import OrderedDict
import numpy as np
import cv2
from utils import ImgUtils


class Tracker:
    MAX_MISSING = 5
    getCentroid = ImgUtils.getCentroid

    rois = OrderedDict()            #rois currently tracked                         {id:roi}
    disappeared = OrderedDict()     #number of frames items have been missing for   {id:num_frames_missing}
    num = 1                         #number of items counted so far                  id

    def __init__(self, maxDisappeared = MAX_MISSING):
        pass

    def startTracking(self, roi):
        print("Start tracking")
        self.rois[self.num] = roi
        self.disappeared[self.num] = 0
        self.num += 1

    def unregister(self, ID):
        print("Forgetting")
        del self.rois[ID]
        del self.disappeared[ID]

    def match(self, newRois):
        if len(self.rois) < 1:
            for roi in newRois:
                self.startTracking(roi)

        detectedCentroids = []              #freshly received centroids
        trackedCentroids  = []              #currently tracked centroids

        for roi in newRois:
            detectedCentroids.append(self.getCentroid(roi))

        for ID, roi in self.rois.items():
            trackedCentroids.append(self.getCentroid(roi))

        D = cdist(np.array(trackedCentroids), np.array(detectedCentroids))
        correspondence = linear_sum_assignment(D)
        print(D)

        roisDIndicesUsed = set()
        roisTIdsUsed = set()


if __name__ == "__name__":
    T = Tracker()
