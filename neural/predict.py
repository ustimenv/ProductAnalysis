
import mxnet as mx
import numpy as np

# from altNet import SSDv1
from brunette import Brunette

class Predictor:
    def __init__(self):
        self.ctx = mx.gpu()
        self.classes = ['1', '2']
        self.Transer = FrameTransformer()
        self.net = Brunette(classes=self.classes)
        self.net.load_parameters('models/netV6-0--15.params', ctx=self.ctx)
        self.net.collect_params()
        self.net.set_nms(nms_thresh=0.1, nms_topk=400, post_nms=20)

    def predict(self, frame, threshold=0.7):
        h, w, _ = frame.shape
        frame = self.Transer.transform(frame)
        allIds, allScores, allBoxes = self.net(frame)
        out = []

        for cid, score, coords in zip(allIds[0], allScores[0], allBoxes[0]):
            cid = cid.asnumpy()[0]
            score = score.asnumpy()[0]
            if cid < 0 or score < threshold:
                continue
            xmin, ymin, xmax, ymax = coords.asnumpy()
            out.append([cid, score, xmin, ymin, xmax, ymax])
        return out

    def getBoxes1(self, frame, threshold=0.2):
        h, w, _ = frame.shape
        frame = self.Transer.transform(frame)
        anchors, cls_preds, box_preds = self.net(frame)
        cls_probs = mx.nd.SoftmaxActivation(mx.nd.transpose(cls_preds, (0, 2, 1)), mode='channel')
        print(cls_probs.shape, anchors.shape, box_preds.shape)
        out = mx.nd.contrib.MultiBoxDetection(cls_probs, box_preds, anchors, force_suppress=True, clip=False)
        print(out)
        output = []
        for det in out:
            cid = int(det[0])
            if cid < 0:
                continue
            score = det[1]
            if score < threshold:
                continue

            scales = [frame.shape[1], frame.shape[0]] * 2
            xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
            output.append([cid, score, xmin, ymin, xmax, ymax])

        return output


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
        # print(image.shape)
        # image = resize(image, dsize=(300, 300))
        image = image[:, :, (2, 1, 0)]
        image = image.astype(np.float32)
        image -= np.array([123, 117, 104])
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :]
        image = mx.nd.array(image, ctx=mx.gpu())
        return image


if __name__ == "__main__":
    pass