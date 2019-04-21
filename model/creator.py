import numpy as np

import utils
from data import dataset_utils,dataset


class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=64,
                 # pos_ratio=0.25, pos_iou_thresh=0.5,
                 # neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 pos_ratio=0.5, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.3, neg_iou_thresh_lo=0.1
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE: py-faster-rcnn默认的值是0.1

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean,
                 loc_normalize_std):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_bbox, _ = bbox.shape

        # roi是rpn网络生成的候选区域
        # bbox是ground truth
        # # 这里要注意的是bbox区域也可以作为训练样本，所以这里将roi和bbox concat起来
        # roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)  # 采样的正样本数量
        # 每个roi对应每个bbox的IOU
        iou = utils.bbox_iou(roi, bbox)

        # 每个roi对应iou最大的bbox的index
        gt_assignment = iou.argmax(axis=1)
        # 每个roi对应最大iou的值
        max_iou = iou.max(axis=1)

        # 每个roi的label,0为背景类,所以别的类别都+1
        gt_roi_label = label[gt_assignment] + 1

        # 到这里我们得到了所有roi(包括bbox)的最大IOU值和他们的类别标签
        # 在标注类别标签的时候我们并不关心它与bbox最接近

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        # 在大于IOU阈值的roi中选取正样本，为什么正样本比例设的这么低???,只有0.25
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        # print(len(pos_index))
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.shape[0]))
        if pos_index.shape[0] > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)
        # print(len(pos_index))

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        # 在IOU区间内选择负样本,这里的iou区间是[0-0.5],我觉得0.5还挺高的???
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        # print(len(neg_index))
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)
        # print(len(neg_index))

        # The indices that we're selecting (both positive and negative).
        # 正类保留分类,负类标签置0
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[len(pos_index):] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        # 计算4个修正量作为位置回归的ground truth
        gt_roi_loc = utils.bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc - loc_normalize_mean) / loc_normalize_std

        # # debug
        # print('debug')
        # # gt_roi_loc_ = utils.loc2bbox(sample_roi,gt_roi_loc)
        # # print(gt_roi_loc_.shape)
        # # print(gt_roi_loc_[:10])
        # gt_roi_label_ = gt_roi_label - 1
        # # gt_rpn_loc_ = gt_rpn_loc_[gt_rpn_label.numpy()>=0]
        # # dataset_utils.draw_pic(img[0].numpy(),dataset.VOC_BBOX_LABEL_NAMES,gt_rpn_loc_,)
        # # gt_roi_loc_ = gt_roi_loc_[gt_roi_label_>=0]
        # img_ = dataset_utils.inverse_normalize(img[0].numpy())
        # pos_roi_ = sample_roi[:len(pos_index)]
        # pos_roi_cls_ = gt_roi_label_[:len(pos_index)]
        # print(pos_roi_.shape,gt_roi_loc[:len(pos_index)].shape)
        # pos_bbox = utils.loc2bbox(pos_roi_,gt_roi_loc[:len(pos_index)])
        # dataset_utils.draw_pic(img_, dataset.VOC_BBOX_LABEL_NAMES, pos_bbox,pos_roi_cls_)

        # 这里似乎并不能保证选取出来的sample_roi数目一定是128个,因为极端情况下可以有很多不符合条件的roi,即不能选作正样本也不能选做负样本
        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=128,
                 # pos_iou_thresh=0.6, neg_iou_thresh=0.2,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.1,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.
        采样n_sample个anchor,打上[0,1]标签,未被采样到的anchor打-1标签

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size

        n_anchor = len(anchor)
        # inside_index:位置不出界的合法anchor的index
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor_ = anchor[inside_index]
        # print(anchor[:10])

        # argmax_ious:与anchor最接近的bbox索引
        #   比如：anchor0与bbox2最接近,anchor1与bbox0最接近...
        #   那么argmax_ious = [2,0,...]
        # label:大于IOU阈值的标记为正样本1,小于IOU阈值的标记为负样本0,还有一部分被丢弃掉的样本标记为-1
        argmax_ious, label = self._create_label(anchor_, bbox)

        # compute bounding box regression targets
        # 将位置框转换为修正值，作为回归目标
        loc = utils.bbox2loc(anchor_, bbox[argmax_ious])

        # map up to original set of anchors
        # 由于RPN网络生成了所有anchor(9*hh*ww个)的预测,所以这里也生成所有anchor的目标
        # 所以要unmap回原来的数据中,不合法和不需要的样本通过将其label=-1的方式标注出来
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)      # 这里loc填充什么无所谓，最后计算loss的时候会只计算label=1的样本，其余样本的权重weight=0

        return loc, label

    def _create_label(self, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((anchor.shape[0],), dtype=np.int32)
        label.fill(-1)

        # argmax_ious : 每个anchor,与其最接近的bbox的索引,[nanchor,]
        # max_ious : 每个anchor,与其最接近的bbox的IOU值,[nanchor,]
        # gt_argmax_ious : 与bbox最接近的anchor的索引，每个bbox可能对应多个，所以数量不定
        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox,)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        # 这一步是否多余?如果gt_argmax_ious都不被包含在max_ious >= self.pos_iou_thresh里,
        # 就说明最大的IOU值都不满足大于阈值的条件,label是否还应该被设为1？
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        # print(np.sum(label==1),np.sum(label==0))

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox,):
        # ious between the anchors and the gt boxes
        # 这里输入的anchor已经删除了超出图像边界的样本
        ious = utils.bbox_iou(anchor, bbox)                          # [nanchor,nbbox],以下的最接近表示IOU最大
        argmax_ious = ious.argmax(axis=1)                            # 每个anchor,与其最接近的bbox的索引,[nanchor,]
        max_ious = ious[np.arange(ious.shape[0]), argmax_ious]       # 每个anchor,与其最接近的bbox的IOU值,[nanchor,]
        gt_argmax_ious = ious.argmax(axis=0)                         # 每个bbox,与其最接近的anchor的索引,[nbbox,]
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] # 每个bbox,与其最接近的anchor的IOU值[nbbox,]
        # 注意,这里得到的 与每个bbox最近的anchor索引和IOU值 并不是全部的，因为np.argmax只会选取第一个碰到的最大值，所以还需要下面这步来得到所有的最大值的index
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]            # 这里并不关心anchor是跟哪个bbox最接近,全都要

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of size count)

    # fill label
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    # fill loc
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    用非极大值抑制的方法合并anchors,选择预测得分最高的anchor作为ROI,

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.8,
                 n_train_post_nms=300,
                 n_test_post_nms=30,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_post_nms = n_train_post_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember faster_rcnn.eval() to set self.parent_model.training = False
        if self.parent_model.training:
            n_post_nms = self.n_train_post_nms
        else:
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # 根据anchor_bbox和他的修正量loc生成修正后的bbox
        # roi:[n_anchor,4]
        roi = utils.loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        # 将roi的边界clip成原始图像边界
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        # 删除长或宽小于minsize的roi
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # 非最大值抑制 合并anchor,选取最大的n_post_nms个作为最终的roi
        keep = utils.nms(np.ascontiguousarray(np.asarray(roi)),scores=score,threshold=self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi

if __name__ == '__main__':
    pass
