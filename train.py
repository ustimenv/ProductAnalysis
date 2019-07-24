from time import time

import mxnet as mx
from gluoncv.loss import SSDMultiBoxLoss
from mxnet import autograd, gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageRecordDataset

from SSDtrans import SSDTrainTransform
from janet import Janet


class Trainer:
    def __init__(self):
            classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
            width = 300
            height = 300
            self.batchSize = 2
            self.ctx = mx.gpu()

            self.net = Janet(classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9'))
            self.net.initialize(ctx=self.ctx)

            #generate anchors now to compute targets later
            x = mx.nd.zeros(shape=(1, 3, 300, 300), ctx=self.ctx)
            with autograd.train_mode():
                _, _, anchors = self.net(x)
            self.transform = SSDTrainTransform(width, height, anchors.as_in_context(mx.cpu()))


            #Data
            trainDataset = ImageRecordDataset(filename='DataX/annoTrainX.rec')
            self.trainData = DataLoader(trainDataset.transform(self.transform),
                                        self.batchSize, shuffle=True, last_batch='rollover')

            #Training stuff
            self.loss = SSDMultiBoxLoss()
            self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd',
                                         {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})

    def train(self):
        num_epochs = 13
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        print('ookoko')
        for epoch in range(num_epochs):
            counter=0
            print('Commencing epoch', epoch)
            tic = time()

            for i, batch in enumerate(self.trainData):
                counter += 1
                with autograd.record():
                    cls_pred, box_pred, anchors = self.net(batch[0].as_in_context(mx.gpu()))
                    clsTargets, bboxTargets = self.transform.generate(cls_pred, batch[1], batch[2])
                    sumLoss, clsLoss, boxLoss = self.loss(cls_pred, box_pred, clsTargets, bboxTargets)

                    if counter % 1 == 0:
                        print('B:{}, Loss:{:.3f}, \nClsLoss:{}, \nBboxLoss:{}\n\n'.format
                              (i, mx.nd.mean(sumLoss[0]).asscalar(), clsLoss[0].asnumpy(), boxLoss[0].asnumpy()))

                autograd.backward(sumLoss)
                self.trainer.step(self.batchSize)
                ce_metric.update(0, [l * self.batchSize for l in clsLoss])
                smoothl1_metric.update(0, [l * self.batchSize for l in boxLoss])

            try:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                print('[Epoch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'
                      .format(epoch, self.batchSize / (time() - tic), name1, loss1, name2, loss2))
            except:
                print('unexpecrted error')
                pass
            self.net.save_parameters('params/1X03Xnet'+str(epoch)+'.params')


if __name__ == "__main__":
    T = Trainer()
    T.train()


