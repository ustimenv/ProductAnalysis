from gluoncv.data.transforms import image as timage, bbox as tbbox, experimental
from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
import numpy as np
import mxnet as mx


class SSDTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.3, box_norm=(0.1, 0.1, 0.2, 0.2),
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
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=3, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)
        label = label[4:]

        # random expansion with prob 0.5
        label = np.expand_dims(label, axis=0)
        if False and np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        # h, w, _ = img.shape
        # bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        # x0, y0, w, h = crop
        # img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        # h, w, _ = img.shape
        # interp = np.random.randint(0, 5)
        # img = timage.imresize(img, self._width, self._height, interp=interp)
        # bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
####
        # random horizontal flip
        # h, w, _ = img.shape
        # img, flips = timage.random_flip(img, px=0.5)
        # bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
####
        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)


        # generate training target so cpu workers can help reduce the workload on gpu

        gt_bboxes = mx.nd.array(bbox[:, 1:])
        gt_ids    = mx.nd.array(bbox[:, 0])
        # print(gt_bboxes)
        return img.as_in_context(mx.gpu()), gt_bboxes, gt_ids
        # cls_targets, box_targets, _ = self._target_generator(self._anchors, None, gt_bboxes, gt_ids)
        # return img.as_in_context(mx.gpu()), cls_targets[0].as_in_context(mx.gpu()), box_targets[0].as_in_context(mx.gpu())

    def generate(self, clsPreds, gt_bboxes, gt_ids):
        cls_targets, box_targets, _ = self._target_generator(self._anchors.as_in_context(mx.gpu()),
                                                             clsPreds.as_in_context(mx.gpu()),
                                                             gt_bboxes.as_in_context(mx.gpu()),
                                                             gt_ids.as_in_context(mx.gpu()))
        return cls_targets.as_in_context(mx.gpu()), box_targets.as_in_context(mx.gpu())