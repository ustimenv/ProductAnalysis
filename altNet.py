import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn


class SSDv1(nn.Block):
    sizes = [[0.1, 0.15], [0.2, 0.25], [0.30, 0.34], [0.40, 0.50], [0.55, 0.7]]
    ratios = [[1, 2, 1.4]] * 2 + [[1, 2, 0.8, 3, 0.8]] * 2 + [[1, 2, 1.5]] * 1
    anchorSeed = 3

    num_anchors = len(sizes[0]) + len(ratios[0]) - 1

    def __init__(self, **kwargs):
        super(SSDv1, self).__init__(**kwargs)
        # anchor box sizes and ratios for 5 feature scales.
        self.num_classes = 2

        for i in range(self.anchorSeed):
            # The assignment statement is self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, self.get_blk(i))
            setattr(self, 'cls_%d' % i, self.classPredictor())
            setattr(self, 'bbox_%d' % i, self.boxPredictzor())

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * self.anchorSeed, [None] * self.anchorSeed, [None] * self.anchorSeed
        for i in range(self.anchorSeed):
            # getattr(self, 'blk_%d' % i) accesses self.lk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = self.blk_forward(
                X, getattr(self, 'blk_%d' % i), self.sizes[i], self.ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        # In the reshape function, 0 indicates that the batch size remains
        # unchanged
        return (nd.concat(*anchors, dim=1),
                self.concatPredictions(cls_preds).reshape((0, -1, self.num_classes + 1)),
                self.concatPredictions(bbox_preds))

    def blk_forward(self, X, blk, size, ratio, cls_predictor, bbox_predictor):
        Y = blk(X)
        anchors = mx.contrib.ndarray.MultiBoxPrior(data=Y, sizes=size, ratios=ratio)
        cls_preds = cls_predictor(Y)
        bbox_preds = bbox_predictor(Y)
        return (Y, anchors, cls_preds, bbox_preds)

    def get_blk(self, i):
        if i == 0:
            blk = self.base_net()
        elif i == 2:
            blk = nn.GlobalMaxPool2D()
        else:
            blk = self.downSample(128)
        return blk


    def base_net(self):
        out = nn.Sequential()
        for nfilters in [16, 32, 64]:
            out.add(self.downSample(nfilters))
        return out


    def classPredictor(self):
        return nn.Conv2D(channels=self.num_anchors * (self.num_classes + 1), kernel_size=3, padding=1)

    def boxPredictzor(self):
        return nn.Conv2D(channels=self.num_anchors * 4, kernel_size=3, padding=1)

    def flattenPrediction(self, pred):
        return pred.transpose(axes=(0, 2, 3, 1)).flatten()

    def concatPredictions(self, preds):
        return nd.concat(*[self.flattenPrediction(p) for p in preds], dim=1)

    def downSample(self, num_channels):
        out = nn.Sequential()
        for _ in range(2):
            out.add(
                nn.Conv2D(channels=num_channels, kernel_size=3, strides=1, padding=1),
                nn.BatchNorm(),
                nn.Activation("relu")
                   )
        out.add(nn.MaxPool2D(2))
        return out





