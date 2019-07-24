import gc
from gluoncv.data.transforms import image as timage

from janet import *
from mxnet import profiler


class Predictor:
    def __init__(self):
        self.ctx = mx.gpu()
        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.Transer = FrameTransformer()
        self.net = Janet(classes=classes)
        self.net.set_nms(nms_thresh=0.45, nms_topk=100, post_nms=20)
        self.net.load_parameters('params/1X02Xnet12.params', ctx=self.ctx)
        self.net.collect_params()
        gc.collect()



    def getBoxes(self, frame, threshold=0.2):
        h, w, _ = frame.shape
        frame = self.Transer.transform(mx.ndarray.array(frame, ctx=mx.gpu()))

        allIds, allScores, allBoxes = self.net(frame)
        out = []
        w = 300
        h = 300
        offsetX = 0
        offsetY = 0

        for cid, score, coords in zip(allIds[0], allScores[0], allBoxes[0]):
            cid = cid.asnumpy()[0]
            score = score.asnumpy()[0]
            if cid < 0 or score < threshold:
                continue
            xmin, ymin, xmax, ymax = coords.asnumpy()
            out.append((cid, score, xmin * w + offsetX / 2, ymin * h + offsetY / 2, xmax * w + offsetX / 2,
                        ymax * h + offsetY / 2))
        gc.collect()
        return out


class FrameTransformer:
    def __init__(self, width=300, height=300, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def transform(self, img):
        # h, w, _ = src.shape
        # img = timage.imresize(src, self._width, self._height, interp=9)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img


if __name__ == "__main__":
    pass