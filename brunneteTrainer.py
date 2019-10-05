from time import time

from gluoncv.loss import SSDMultiBoxLoss
from mxnet import gluon, autograd
from mxnet import image

from targetGen import *
from brunette import Brunette


class Trainer:
    BATCH_SIZE = 4
    NUM_EPOCHS = 13
    classes = ['1', '2', '3']
    ctx = mx.gpu()
    lossFunction = SSDMultiBoxLoss()

    def __init__(self):
        self.net = Brunette(classes=self.classes, use_bn=True)
        self.net.initialize(ctx=self.ctx)

        self.trainIter = image.ImageDetIter(batch_size=self.BATCH_SIZE, data_shape=(3, 300, 300),
                                            path_imgrec='utils/Trainx.rec',
                                            path_imgidx='utils/Trainx.idx',
                                            # path_img='/home/vlad/Work/1/MLworkDir/Trainx.idx',
                                            )

        with autograd.train_mode():
            _, _, anchors = self.net(mx.ndarray.zeros(shape=(self.BATCH_SIZE, 3, 300, 300), ctx=self.ctx))

        self.T = TargetGenV2(anchors=anchors.as_in_context(mx.cpu()), height=300, width=300)

        self.net.collect_params().reset_ctx(self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

    def train(self):
        print('Preparing to train')
        metricClass = mx.metric.Loss('CrossEntropy')
        metricBox = mx.metric.Loss('SmoothL1')

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

                    sumLoss, clsLoss, bboxLoss = self.lossFunction(clsPreds,
                                                                   bboxPreds,
                                                                   clsTargets.as_in_context(self.ctx),
                                                                   bboxTargets.as_in_context(self.ctx))
                    '''Compute Losses'''
                    if (i+1) % 20 == 0:
                        print('B:{}, Loss:{:.3f}, \nClsLoss:{}, \nBboxLoss:{}\n\n'.format
                              (i, mx.nd.mean(sumLoss[0]).asscalar(), clsLoss[0].asnumpy(), bboxLoss[0].asnumpy()))
                    # back-propagate #TODO 1st vs 2nd order derivative??
                    autograd.backward(sumLoss)
                self.trainer.step(self.BATCH_SIZE)

                metricClass.update(0, [l * self.BATCH_SIZE for l in clsLoss])
                metricBox.update(0, [l * self.BATCH_SIZE for l in bboxLoss])

            name1, loss1 = metricClass.get()
            name2, loss2 = metricBox.get()

            print('[Epoch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'
                  .format(epoch, self.BATCH_SIZE / (time() - tic), name1, loss1, name2, loss2))

            self.net.save_parameters('models/' + 'netV1-' + str(epoch) + '.params')


if __name__ == "__main__":
    T = Trainer()
    T.train()

