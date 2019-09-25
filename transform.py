import cv2


class Transformer:
    detectionRoi = [0, 0, 0, 0]
    transform = None
    resize = None

    def __init__(self, name):
        self.transform = eval('self.'+name+'Transform')
        self.resize = eval('self.'+name+'Resize')

    def raw1Transform(self, img):
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        contrast = cv2.bitwise_not(contrast)
        contrast = cv2.boxFilter(src=contrast, ddepth=-1, ksize=(3, 17))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        contrast = cv2.morphologyEx(contrast, cv2.MORPH_DILATE, kernel, iterations=2)
        contrast = cv2.threshold(src=contrast, maxval=255, thresh=200, type=cv2.THRESH_BINARY)[1]
        return contrast

    def raw1Resize(self, img, ymin=300, ymax=670, xmin=250, xmax=760):
        return img[ymin:ymax, xmin:xmax]

    def postbake1Transform(self, img):
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
        contrast = cv2.medianBlur(contrast, 7, 5)
        contrast = cv2.threshold(src=contrast, maxval=255, thresh=70, type=cv2.THRESH_BINARY)[1]
        return contrast

    def postbake1Resize(self, img, ymin=150, ymax=1000, xmin=530, xmax=830):
        return img[ymin:ymax, xmin:xmax]

    def transformMorph(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        img = cv2.GaussianBlur(img, (45, 5), 0)
        contrast = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        return contrast

    def correctLighting(self, img):
        cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_BGR2YUV)
        channels = cv2.split(img)
        cv2.equalizeHist(channels[0], channels[0])
        cv2.merge(channels, img)
        cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_YUV2BGR)
        return img

