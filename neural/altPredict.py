import cv2
import mxnet as mx
import numpy as np
from altNet import SSDv1


class Predictor:
    def __init__(self):
        self.ctx = mx.gpu()
        self.classes = ['1', '2']
        self.net = SSDv1()
        self.net.load_parameters('models/netV5-1--7.params', ctx=self.ctx)
        self.net.collect_params()

    def predict(self, img, threshold=0.72):
        h, w, _ = img.shape
        frame = self.transform(img)
        # _, _, h, w = frame.shape
        anchors, clsPreds, boxPreds = self.net(frame)
        clsProbs = mx.nd.SoftmaxActivation(mx.nd.transpose(clsPreds, (0, 2, 1)), mode='channel')
        rawOuput = mx.nd.contrib.MultiBoxDetection(*[clsProbs, boxPreds, anchors], force_suppress=True, clip=False)
        out = []
        for x in rawOuput[0].asnumpy():
            cid = x[0]
            prob = x[1]
            if cid == -1 or prob < threshold:
                continue
            box = [x[2]*w, x[3]*h, x[4]*w, x[5]*h]
            # print(x[2]*w, x[3]*h, x[4]*w, x[5]*h)
            # box = [x[2], x[3], x[4], x[5]]

            out.append([cid, prob, box])
        return out


    def transform(self, image):
        image = image[:, :, (2, 1, 0)]
        image = cv2.resize(image, (300, 300))
        image = image.astype(np.float32)
        image -= np.array([123, 117, 104])
        image = np.transpose(image, (2, 0, 1))
        image = image[np.newaxis, :]
        image = mx.nd.array(image, ctx=mx.gpu())
        return image


if __name__ == "__main__":
    pass