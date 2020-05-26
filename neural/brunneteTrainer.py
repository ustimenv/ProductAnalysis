import argparse
from time import time

from gluoncv.loss import SSDMultiBoxLoss
from mxnet import gluon, autograd
from mxnet import image

from targetGen import TargetGenV2
import mxnet as mx
from brunette import Brunette


class Trainer:
    NUM_EPOCHS = 20
    classes = ['1', '2']
    ctx = mx.gpu()
    lossFunction = SSDMultiBoxLoss()

    def __init__(self, batchSize):
        self.net = Brunette(classes=self.classes)
        self.net.initialize(mx.init.Xavier(magnitude=2), ctx=self.ctx)
        self.batchSize = batchSize
        self.trainIter = image.ImageDetIter(batch_size=self.batchSize, data_shape=(3, 300, 300),
                                            path_imgrec='utils/TrainY.rec',
                                            path_imgidx='utils/TrainY.idx',
                                            shuffle=True, mean=True,
                                            brightness=0.3, contrast=0.3, saturation=0.3, pca_noise=0.3, hue=0.3
                                            )

        # with autograd.train_mode():
        #     _, _, anchors = self.net(mx.ndarray.zeros(shape=(self.batchSize, 3, 300, 300), ctx=self.ctx))

        # self.T = TargetGenV2(anchors=anchors.as_in_context(mx.cpu()), height=300, width=300)
        print("2")
        self.net.collect_params().reset_ctx(self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 3e-4})

    def training_targets(self, anchors, class_preds, labels):
        class_preds = class_preds.transpose(axes=(0, 2, 1))  # batchsize x num_cls x num_anchors
        box_target, box_mask, cls_target = mx.contrib.ndarray.MultiBoxTarget(anchors, labels, class_preds,
                                                            overlap_threshold=.5,
                                                            ignore_label=-1, negative_mining_ratio=3,
                                                            minimum_negative_samples=0,
                                                            negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
                                                            name="multibox_target")

        return box_target, box_mask, cls_target

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
                    clsPreds, bboxPreds, anchors = self.net(X)
                    # clsTargets, bboxTargets = self.T.generateTargets(Y, clsPreds)
                    clsTargets, _,  bboxTargets = self.training_targets(anchors, clsPreds, Y)
                    print(clsTargets, bboxTargets)
                    return
                    sumLoss, clsLoss, bboxLoss = self.lossFunction(clsPreds,
                                                                   bboxPreds,
                                                                   clsTargets.as_in_context(self.ctx),
                                                                   bboxTargets.as_in_context(self.ctx))

                    '''Compute Losses'''
                    if (i+1) % 200 == 0:
                        print('B:{}, Loss:{:.3f}, \nClsLoss:{}, \nBboxLoss:{}\n\n'.format
                              (i, mx.nd.mean(sumLoss[0]).asscalar(), clsLoss[0].asnumpy(), bboxLoss[0].asnumpy()))
                    # back-propagate #TODO 1st vs 2nd order derivative??
                    autograd.backward(sumLoss)
                self.trainer.step(self.batchSize)

                metricClass.update(0, [l * self.batchSize for l in clsLoss])
                metricBox.update(0, [l * self.batchSize for l in bboxLoss])

            name1, loss1 = metricClass.get()
            name2, loss2 = metricBox.get()

            print('[Epoch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'
                  .format(epoch, self.batchSize / (time() - tic), name1, loss1, name2, loss2))
            if epoch % 2 != 0:
                self.net.save_parameters('models/' + 'netV6-0--' + str(epoch) + '.params')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', dest='batchSize', type=int, default=2)
    args = parser.parse_args()
    batchSize = args.batchSize
    T = Trainer(batchSize)
    T.train()

