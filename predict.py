import gc

import numpy as np

from brunette import *


class Predictor:
    def __init__(self):
        self.ctx = mx.gpu()
        self.classes = ['1', '2', '3']
        self.Transer = FrameTransformer()
        self.net = Brunette(classes=self.classes)
        self.net.load_parameters('models/netV31-8.params', ctx=self.ctx)

        self.net.collect_params()
        gc.collect()

    def getBoxes(self, frame, threshold=0.2):
        h, w, _ = frame.shape
        frame = self.Transer.transform(frame)
        allIds, allScores, allBoxes = self.net(frame)
        out = []
        # print(h, w)
        w = 300
        h = 300
        xOffset = 0#w - 300
        yOffset = 0#h - 300

        for cid, score, coords in zip(allIds[0], allScores[0], allBoxes[0]):
            cid = cid.asnumpy()[0]
            score = score.asnumpy()[0]
            if cid < 0 or score < threshold:
                continue
            xmin, ymin, xmax, ymax = coords.asnumpy()
            xmin *= w; ymin *= h; xmax *= w; ymax *= h
            out.append([cid, score, xmin, ymin, xmax, ymax])
        gc.collect()
        return out


class FrameTransformer:
    def __init__(self, width=300, height=300):
        self._width = width
        self._height = height
        self._mean = [123.68/300, 116.28/300, 103.53/300]
        self._std = [58.395/300, 57.12/300, 57.375/300]

    def transform0(self, image):
        image = image[:, :, (2, 1, 0)]
        image = image.astype(np.float32)
        image -= np.array([123, 117, 104])
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :]
        image = mx.nd.array(image, ctx=mx.gpu())
        return image

    def transform(self, image):
        from cv2 import resize
        image = resize(image, (300, 300))
        image = image[:, :, (2, 1, 0)]
        image = image.astype(np.float32)
        image -= np.array([123, 117, 104])
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :]
        image = mx.nd.array(image, ctx=mx.gpu())
        return image



if __name__ == "__main__":
    pass