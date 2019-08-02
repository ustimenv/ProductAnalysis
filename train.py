from time import time

from gluoncv.loss import SSDMultiBoxLoss
from mxnet import gluon
from mxnet import image

from janet import *
from targetGen import *


class Trainer:
    BATCH_SIZE = 16
    NUM_EPOCHS = 13
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    ctx = mx.gpu()
    mboxLoss = SSDMultiBoxLoss()

    def __init__(self):
        self.net = JanetRes(classes=self.classes, use_bn=True)
        self.net.initialize(ctx=self.ctx)

        self.trainIter = image.ImageDetIter(batch_size=self.BATCH_SIZE, data_shape=(3, 300, 300),
                                            path_imgrec='../DataX/annoTrainX.rec',
                                            path_imgidx='../DataX/annoTrainX.idx',
                                            path_imglist='../DataX/annoTrainX.lst',
                                            path_root='../DataX/', shuffle=True, mean=True,
                                            brightness=0.3, contrast=0.3, saturation=0.3, pca_noise=0.3, hue=0.3)

        with autograd.train_mode():
            _, _, anchors = self.net(mx.ndarray.zeros(shape=(self.BATCH_SIZE, 3, 300, 300), ctx=self.ctx))
        self.T = TargetGenV1(anchors=anchors.as_in_context(mx.cpu()), height=300, width=300)

        self.net.collect_params().reset_ctx(self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

    def train(self):
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')
        for epoch in range(self.NUM_EPOCHS):
            print('Commencing epoch', epoch)
            tic = time()
            self.trainIter.reset()

            for i, batch in enumerate(self.trainIter):
                X = batch.data[0].as_in_context(self.ctx)
                Y = batch.label[0].as_in_context(self.ctx)

                with autograd.record():
                    '''Make Predictions'''
                    clsPreds, bboxPreds, _ = self.net(X)
                    clsTargets, bboxTargets = self.T.generateTargets(Y, clsPreds)

                    sumLoss, clsLoss, bboxLoss = self.mboxLoss(clsPreds,
                                                               bboxPreds,
                                                               clsTargets.as_in_context(self.ctx),
                                                               bboxTargets.as_in_context(self.ctx))
                    '''Compute Losses'''
                    if (i+1) % 200 == 0:
                        print('B:{}, Loss:{:.3f}, \nClsLoss:{}, \nBboxLoss:{}\n\n'.format
                              (i, mx.nd.mean(sumLoss[0]).asscalar(), clsLoss[0].asnumpy(), bboxLoss[0].asnumpy()))

                    autograd.backward(sumLoss)
                self.trainer.step(self.BATCH_SIZE)
                ce_metric.update(0, [l * self.BATCH_SIZE for l in clsLoss])
                smoothl1_metric.update(0, [l * self.BATCH_SIZE for l in bboxLoss])

            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            print('[Epoch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'
                  .format(epoch, self.BATCH_SIZE / (time() - tic), name1, loss1, name2, loss2))

            self.net.save_parameters('../params/2X7' + 'Xnet' + str(epoch) + '.params')


if __name__ == "__main__":
    T = Trainer()
    T.train()

