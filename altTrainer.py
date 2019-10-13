import argparse
from time import time

import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon, autograd, image
from mxnet import nd
from mxnet.initializer import Xavier

from altNet import SSDv1


class SSDtrainer:
    ctx = mx.gpu()
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    bbox_loss = gluon.loss.L1Loss()

    def __init__(self, batchSize):
        self.net = SSDv1()
        self.net.initialize(init=Xavier(), ctx=self.ctx)
        self.batchSize = batchSize
        self.trainIter = image.ImageDetIter(batch_size=self.batchSize, data_shape=(3, 300, 300),
                                            path_imgrec='utils/TrainY.rec', path_imgidx='utils/TrainY.idx',)

        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

    def calc_loss(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        cls = self.cls_loss(cls_preds, cls_labels)
        bbox = self.bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
        return cls + bbox

    def cls_eval(self, cls_preds, cls_labels):
        return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()

    def bbox_eval(self, bbox_preds, bbox_labels, bbox_masks):
        return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()

    def transformer(self, img, label):
        yl, xl, _ = img.shape
        img = img.asnumpy()
        if xl < 300:
            img = cv2.copyMakeBorder(img, 0, 0, int((300 - xl) / 2), int((300 - xl) / 2), cv2.BORDER_REPLICATE)
        if yl < 200:
            img = cv2.copyMakeBorder(img, int((200 - yl) / 2), int((200 - yl) / 2), 0, 0, cv2.BORDER_REPLICATE)
        img = cv2.resize(img, (300, 200))
        return mx.ndarray.array(np.transpose(img, (2, 0, 1)), ctx=self.ctx, dtype=np.float32) / 255, label

    @staticmethod
    def training_targets(default_anchors, class_predicts, labels):
        class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
        z = mx.nd.contrib.MultiBoxTarget(*[default_anchors, labels, class_predicts])
        box_target = z[0]  # box offset target for (x, y, width, height)
        box_mask = z[1]  # mask is used to ignore box offsets we don't want to penalize, e.g. negative samples
        cls_target = z[2]  # cls_target is an array of labels for all anchors boxes
        return box_target, box_mask, cls_target

    def train(self):
        num_epochs = 8

        for epoch in range(num_epochs):
            print("EPOCH ", epoch)
            self.trainIter.reset()
            for i, batch in enumerate(self.trainIter):
                X = batch.data[0].as_in_context(self.ctx)
                Y = batch.label[0].as_in_context(self.ctx)

                with autograd.record():
                    anchors, cls_preds, bbox_preds = self.net(X)
                    # Label the category and offset of each anchor box
                    # print(anchors.shape, '||', Y.shape, '||', cls_preds.transpose((0, 2, 1)).shape)
                    # bbox_labels, bbox_masks, cls_labels = mx.contrib.ndarray.MultiBoxTarget(anchors, Y, cls_preds.transpose((0, 2, 1)))
                    bbox_labels, bbox_masks, cls_labels = self.training_targets(anchors, cls_preds, Y)
                    # print(cls_preds, bbox_preds, anchors)
                    # print(bbox_labels, cls_labels)

                    # Calculate the loss function using the predicted and labeled
                    # category and offset values
                    l = self.calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
                    if i % 100 == 0:
                        print(i, '-->', l)

                l.backward()
                self.trainer.step(self.batchSize)
            self.net.save_parameters('netV5-1--'+str(epoch)+'.params')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', dest='batchSize', type=int, default=2)
    args = parser.parse_args()
    batchSize = args.batchSize

    T = SSDtrainer(batchSize)
    T.train()

