import cv2


class StandardDetector:
    def __init__(self):
        pass

    def detect(self, contrast):
        contours, h = cv2.findContours(contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        currentRois = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) == 0 or w < 40 or h < 40 or w>400 or h>400:
                continue
            #coordinates of the current bounding box
            x1 = x #- 40
            x2 = x1 + w #+ 60
            y1 = y #- 20
            y2 = y1 + h# +50
            currentRois.append((x1, y1, x2, y2))
        return currentRois


