from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
from gluoncv.nn.bbox import BBoxCenterToCorner
from gluoncv.nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from gluoncv.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from gluoncv.nn.sampler import OHEMSampler, NaiveSampler
from mxnet import nd
from mxnet.gluon import Block
import numpy as np
from gluoncv.data.transforms import image as timage, bbox as tbbox, experimental
import mxnet as mx


class TrainTransform(object):
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        if anchors is None:
            return
        # since we do not have predictions yet, so we ignore sampling here
        self._target_generator = SSDTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def transform(self, src, label):
        roi = np.expand_dims(label[0, 1:5], axis=0)
        print(roi, label)
        # random color jittering
        img = experimental.image.random_color_distort(src)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(roi, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, roi

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
        #
        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        bbox = [x/300 for x in bbox[0]]
        label[0, 1:5] = bbox
        print(label)
        return img, bbox#.astype(img.dtype)

    def generateTargets(self, img, label, clsPreds):
        # generate training target so cpu workers can help reduce the workload on gpu
        # gt_bboxes = mx.nd.array(label[np.newaxis, :, 1:4])
        # gt_ids = mx.nd.array(label[np.newaxis, :, 0])
        gt_ids= mx.nd.array(label[:, :, 0:1])
        gt_bboxes= mx.nd.array(label[:, :, 1:5])

        # print(label, gt_bboxes, gt_ids)
        cls_targets, box_targets, _ = self._target_generator(self._anchors, clsPreds, gt_bboxes, gt_ids)
        return cls_targets, box_targets


class TargetGenerator:
    def __init__(self, anchors, iou_thresh=0.5, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(TargetGenerator, self).__init__(**kwargs)
        self.anchors = anchors
        self._matcher = CompositeMatcher(
            [BipartiteMatcher(share_max=False), MaximumMatcher(iou_thresh)])
        if negative_mining_ratio > 0:
            self._sampler = OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
            self._use_negative_sampling = True
        else:
            self._sampler = NaiveSampler()
            self._use_negative_sampling = False
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def forward(self, cls_preds, label):
        """Generate training targets."""
        gt_ids = mx.nd.array(label[:, :, 0:1,])
        gt_bboxes = mx.nd.array(label[:, :, 1:5])

        anchors = self._center_to_corner(self.anchors.reshape((-1, 4)))
        ious = nd.transpose(nd.contrib.box_iou(anchors, gt_bboxes), (1, 0, 2))
        matches = self._matcher(ious)
        if self._use_negative_sampling:
            samples = self._sampler(matches, cls_preds, ious)
        else:
            samples = self._sampler(matches)
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_bboxes)
        return cls_targets, box_targets, box_masks


class TargetGenV1:
    def __init__(self, anchors, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self.anchors = anchors
        iou_thresh = 0.2
        stds = (0.1, 0.1, 0.2, 0.2)
        self.tGen = SSDTargetGenerator(iou_thresh=iou_thresh, stds=stds)

    def generateTargets(self, label, clsPreds, ctx=mx.cpu()):
        # gt_bboxes = mx.nd.array(label[np.newaxis, :, 1:4])
        # gt_ids = mx.nd.array(label[np.newaxis, :, 0])

        gt_ids= mx.nd.array(label[:, :, 0:1], ctx=ctx)
        gt_bboxes= mx.nd.array(label[:, :, 1:5], ctx=ctx)
        gt_ids = mx.nd.softmax(gt_ids, axis=0)
        cls_targets, box_targets, _ = self.tGen(self.anchors, clsPreds.as_in_context(ctx), gt_bboxes, gt_ids)

        return cls_targets, box_targets
