import mxnet as mx
from gluoncv.model_zoo.ssd.target import SSDTargetGenerator


class TargetGenV1:
    def __init__(self, anchors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.anchors = anchors
        iou_thresh = 0.2
        stds = (0.1, 0.1, 0.2, 0.2)
        self.tGen = SSDTargetGenerator(iou_thresh=iou_thresh, stds=stds)

    def generateTargets(self, label, clsPreds, ctx=mx.cpu()):
        gt_ids= mx.nd.array(label[:, :, 0:1], ctx=ctx)
        gt_bboxes= mx.nd.array(label[:, :, 1:5], ctx=ctx)
        # gt_ids = mx.nd.softmax(gt_ids, axis=0)

        cls_targets, box_targets, _ = self.tGen(self.anchors, clsPreds.as_in_context(ctx), gt_bboxes, gt_ids)
        return cls_targets, box_targets


class TargetGenV2:
    def __init__(self, anchors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.anchors = anchors
        iou_thresh = 0.2
        stds = (0.1, 0.1, 0.2, 0.2)
        self.tGen = SSDTargetGenerator(iou_thresh=iou_thresh, stds=stds)

    def generateTargets(self, label, clsPreds, ctx=mx.cpu()):
        gt_ids = mx.nd.array(label[:, :, 0:1], ctx=ctx)
        gt_bboxes = mx.nd.array(label[:, :, 1:5], ctx=ctx)
        # gt_ids = mx.nd.softmax(gt_ids, axis=0)

        cls_targets, box_targets, _ = self.tGen(self.anchors, clsPreds.as_in_context(ctx), gt_bboxes, gt_ids)
        return cls_targets, box_targets

