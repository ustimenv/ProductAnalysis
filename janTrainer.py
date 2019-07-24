from time import time

from gluoncv.data.transforms import experimental
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.loss import SSDMultiBoxLoss
from gluoncv.model_zoo.ssd.target import SSDTargetGenerator

from mxnet import autograd, gluon
from mxnet import image
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageRecordDataset
from mxnet.initializer import Xavier
from janet import *
import mxnet as mx
import numpy as np
from targetGen import *
import matplotlib.pyplot as plt
import cv2

class JanTrainer:
    batchSize = 2
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    ctx = mx.gpu()
    mboxLoss = SSDMultiBoxLoss()

    def __init__(self):
        self.net = Janet(classes=self.classes)
        self.net.initialize(init=Xavier(), ctx=self.ctx)
        self.net.collect_params()
        self.trainIter = image.ImageDetIter(batch_size=self.batchSize, data_shape=(3, 300, 300),
                                            path_imgrec='DataX/annoTrainX.rec',
                                            path_imgidx='DataX/annoTrainX.idx',
                                            path_imglist='DataX/annoTrainX.lst',
                                            path_root='DataX/', shuffle=False)

        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})
        with autograd.train_mode():
            _, _, anchors = self.net(mx.ndarray.zeros(shape=(4, 3, 300, 300), ctx=self.ctx))
        self.T = TargetGenV1(anchors=anchors.as_in_context(mx.cpu()), height=300, width=300)

    def train(self):
        num_epochs = 5
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')
        print('ookoko')

        for epoch in range(num_epochs):
            counter=0
            print('Commencing epoch', epoch)
            tic = time()
            self.trainIter.reset()

            for i, batch in enumerate(self.trainIter):
                counter+=1
                X = batch.data[0].as_in_context(self.ctx)
                Y = batch.label[0].as_in_context(self.ctx)

                with autograd.record():
                    '''Make Predictions'''
                    dummy3 = mx.ndarray.zeros((1,2,3), ctx=mx.cpu())

                    clsPreds, bboxPreds, _ = self.net(X)
                    dummy3 = mx.ndarray.zeros((1,2,3), ctx=mx.cpu())

                    clsTargets, bboxTargets = self.T.generateTargets(Y, mx.nd.softmax(clsPreds,axis=0))
                    dummy3 = mx.ndarray.zeros((1,2,3), ctx=mx.cpu())

                    sumLoss, clsLoss, bboxLoss = self.mboxLoss(clsPreds, bboxPreds, clsTargets.as_in_context(self.ctx), bboxTargets.as_in_context(self.ctx))
                    dummy3 = mx.ndarray.zeros((1,2,3), ctx=mx.cpu())

                    # Y = Y.asnumpy()
                    # gtCids = mx.nd.array(Y[:, :, 0:1,])
                    # gtBoxes = mx.nd.array(Y[:, :, 1:5])
                    '''Compute Losses'''
                    if counter % 200 == 0:
                        print('B:{}, Loss:{:.3f}, \nClsLoss:{}, \nBboxLoss:{}\n\n'.format
                              (i, mx.nd.mean(sumLoss[0]).asscalar(), clsLoss[0].asnumpy(), bboxLoss[0].asnumpy()))

                autograd.backward(sumLoss)
                self.trainer.step(self.batchSize)
                ce_metric.update(0, [l * self.batchSize for l in clsLoss])
                smoothl1_metric.update(0, [l * self.batchSize for l in bboxLoss])
            try:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                print('[Epoch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'
                      .format(epoch, self.batchSize / (time() - tic), name1, loss1, name2, loss2))
            except:
                print('unexpecrted eror')
                pass

            self.net.save_parameters('params/1X01Xnet'+str(epoch)+'.params')


    #
    # def trainL(self):
    #     num_epochs = 5
    #     ce_metric = mx.metric.Loss('CrossEntropy')
    #     smoothl1_metric = mx.metric.Loss('SmoothL1')
    #     print('uwu')
    #     for epoch in range(num_epochs):
    #         counter=0
    #         print('Commencing epoch', epoch)
    #         tic = time()
    #         self.trainIter.reset()
    #
    #         with autograd.train_mode():
    #             _, _, anchors = self.net(mx.nd.zeros((self.batchSize, 3, 300, 300), ctx=self.ctx))
    #
    #         for i, batch in enumerate(self.trainIter):
    #             counter+=1
    #             # X = batch.as_in_context(self.ctx)
    #             # Y = np.zeros(shape=(self.batchSize, 1, 5))
    #             # V = TrainTransform(300, 300, anchors.as_in_context(mx.cpu()))
    #
    #             # for i in range(self.batchSize):
    #             #     Y[i] = self.parseLabel(label[i])
    #
    #             X = batch.data[0].as_in_context(self.ctx)
    #             Y = batch.label[0]#.as_in_context(self.ctx)
    #             V = TrainTransform(300, 300, anchors.as_in_context(mx.cpu()))
    #             with autograd.record():
    #                 '''Make Predictions'''
    #                 # X, label = V(X, label)
    #                 # print(X.shape, label)
    #                 # print(nd.expand_dims(Y[0], 0))
    #                 # labelT = mx.nd.zeros((self.batchSize, 1, 5))
    #                 # print('Y', Y.shape)
    #
    #                 # X = mx.nd.zeros((self.batchSize, 3, 300, 300))
    #                 # for i in range(self.batchSize):
    #                 #     X[i], Y[i] = V.transform(batch[i], Y[i])
    #
    #                 clsPreds, bboxPreds, anchors = self.net(X)
    #                 # print(clsPreds, bboxPreds)
    #
    #                 clsTargets, bboxTargets = V.generateTargets(X.as_in_context(mx.cpu()), Y, clsPreds.as_in_context(mx.cpu()))
    #                 # print(clsTargets, mx.nd.softmax(clsTargets, axis=1))
    #
    #                 sumLoss, clsLoss, bboxLoss = self.mboxLoss(clsPreds,
    #                                                            bboxPreds,
    #                                                            mx.nd.softmax(clsTargets, axis=1).as_in_context(self.ctx),
    #                                                            bboxTargets.as_in_context(self.ctx))
    #
    #                 '''Compute Losses'''
    #                 if counter % 1 == 0:
    #                     print('B:{}, Loss:{:.3f}, \nClsLoss:{}, \nBboxLoss:{}\n\n'.format
    #                           (i, mx.nd.mean(sumLoss[0]).asscalar(), clsLoss[0].asnumpy(), bboxLoss[0].asnumpy()))
    #
    #             autograd.backward(sumLoss)
    #             self.trainer.step(self.batchSize)
    #             ce_metric.update(0, [l * self.batchSize for l in clsLoss])
    #             smoothl1_metric.update(0, [l * self.batchSize for l in bboxLoss])
    #         try:
    #             name1, loss1 = ce_metric.get()
    #             name2, loss2 = smoothl1_metric.get()
    #             print('[Epoch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'
    #                   .format(epoch, self.batchSize / (time() - tic), name1, loss1, name2, loss2))
    #         except:
    #             print('unexpecrted eror')
    #             pass
    #
    #         self.net.save_parameters('params/0X13Xnet'+str(epoch)+'.params')
    #         self.net.features.save_parameters('params/0X13Xfeatures'+str(epoch)+'.params')
    #
    # def parseLabel(self, label):
    #     if isinstance(label, nd.NDArray):
    #         label = label.asnumpy()
    #     raw = label.ravel()
    #     if raw.size < 7:
    #         raise RuntimeError("Label shape is invalid: " + str(raw.shape))
    #     header_width = int(raw[0])
    #     obj_width = int(raw[1])
    #     if (raw.size - header_width) % obj_width != 0:
    #         msg = "Label shape %s inconsistent with annotation width %d." \
    #               % (str(raw.shape), obj_width)
    #         raise RuntimeError(msg)
    #     out = np.reshape(raw[header_width:], (-1, obj_width))
    #     # remove bad ground-truths
    #     valid = np.where(np.logical_and(out[:, 3] > out[:, 1], out[:, 4] > out[:, 2]))[0]
    #     if valid.size < 1:
    #         raise RuntimeError('Encounter sample with no valid label.')
    #     return out[valid, :]


if __name__ == "__main__":
    T = JanTrainer()
    T.train()


