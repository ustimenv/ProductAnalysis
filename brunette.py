import mxnet as mx
from gluoncv.model_zoo import SSD
from gluoncv.model_zoo.ssd.anchor import SSDAnchorGenerator
from gluoncv.nn.coder import NormalizedBoxCenterDecoder, MultiPerClassDecoder
from gluoncv.nn.predictor import ConvPredictor
from mxnet import autograd
from mxnet.gluon import nn

from jafeat import JafeatVgg

class Brunette(nn.HybridBlock):
    def __init__(self, classes,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400, post_nms=100,
                 anchor_alloc_size=128, ctx=mx.gpu(), **kwargs):

        super(Brunette, self).__init__(**kwargs)
        self.classes = classes
        im_size = (300, 300)

        # network = 'resnet101_v2'
        # features = ['stage3_activation22', 'stage4_activation2']
        # channels = [512, 512, 256, 256]
        # sizes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
        # ratios = [[1, 2, 1.4]] * 2 + [[1, 2, 0.8, 3, 0.8]] * 2 + [[1, 2, 1.5]] * 2
        # steps = [40 / 300, 100 / 300, 120 / 300, 150 / 300, 180 / 300, 250 / 300]
        # num_layers = len(features) + len(channels)
        #
        # self.features = JafeatVgg()
        # # sizes = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
        # sizes = [30/300, 60/300, 111/300, 162/300, 213/300, 264/300, 315/300]
        # ratios = [[1, 2, 1.4]] * 2 + [[1, 2, 0.8, 3, 0.8]] * 2 + [[1, 2, 1.5]]*2
        # steps = [40 / 300, 100 / 300, 120/300,  150 / 300, 180 / 300, 250 / 300]
        # num_layers = len(ratios)
        #
        # sizes = list(zip(sizes[:-1], sizes[1:]))
        # self._num_layers = num_layers
        # self.classes = classes
        # self.nms_thresh = nms_thresh
        # self.nms_topk = nms_topk
        # self.post_nms = post_nms
        #
        # with self.name_scope():
        #     self.class_predictors = nn.HybridSequential()
        #     self.box_predictors = nn.HybridSequential()
        #     self.anchor_generators = nn.HybridSequential()
        #     asz = anchor_alloc_size
        #     for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
        #         anchor_generator = SSDAnchorGenerator(i, im_size, s, r, st, (asz, asz))
        #         self.anchor_generators.add(anchor_generator)
        #         asz = max(asz // 2, 16)                     # pre-compute larger than 16x16 anchor map
        #         num_anchors = anchor_generator.num_depth
        #         self.class_predictors.add(ConvPredictor(num_anchors * (len(self.classes) + 1)))
        #         self.box_predictors.add(ConvPredictor(num_anchors * 4))
        #     self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
        #     self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)
        ########################################################################
        self.features = JafeatVgg()

        #ratios = [[1, 2, 1.4]] * 2 + [[1, 2, 0.8, 3, 0.8]] * 2 + [[1, 2, 1.5]]*2
        sizes = [21, 45, 99, 153, 207, 261, 315],
        ratios = [[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 3 + [[1, 2, 0.5]] * 2,
        steps = [8, 16, 32, 64, 100, 300],

        num_layers = len(ratios)
        sizes = list(zip(sizes[:-1], sizes[1:]))
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        with self.name_scope():
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            asz = anchor_alloc_size
            for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
                anchor_generator = SSDAnchorGenerator(i, im_size, s, r, st, (asz, asz))
                self.anchor_generators.add(anchor_generator)
                asz = max(asz // 2, 16)                     # pre-compute larger than 16x16 anchor map
                num_anchors = anchor_generator.num_depth
                self.class_predictors.add(ConvPredictor(num_anchors * (len(self.classes) + 1)))
                self.box_predictors.add(ConvPredictor(num_anchors * 4))
            self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

    @property
    def num_classes(self):
        return len(self.classes)

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def hybrid_forward(self, F, x, **kwargs):
        """Hybrid forward"""
        features = self.features(x)
        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]
        box_preds = [F.flatten(F.transpose(bp(feat), (0, 2, 3, 1)))
                     for feat, bp in zip(features, self.box_predictors)]
        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        if autograd.is_training():
            return [cls_preds, box_preds, anchors]
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes
if __name__ == "__main__":
    B = Brunette(['1', '2'])
    print(B)