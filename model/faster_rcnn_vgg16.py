import torch
from torchvision.models import vgg16_bn
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import utils
from model.regin_proposal_network import RegionProposalNetwork
from model.creator import ProposalTargetCreator,AnchorTargetCreator
from collections import namedtuple
import time

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

class ROIHead(nn.Module):
    def __init__(self,classifier,n_class=21,roi_size=(7,7),feat_stride=16):
        super(ROIHead,self).__init__()
        self.feat_stride = feat_stride
        # roi层原理和SSPNet类似,可以把不同大小的图片pooling成大小相同的矩阵
        self.roi_pooling = nn.AdaptiveMaxPool2d(roi_size)
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        self.normal_init(self.cls_loc, 0, 0.001)
        self.normal_init(self.score, 0, 0.01)

    def normal_init(self,m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

    def get_roi_feature(self,x, rois):
        # 获取roi对应feature map上的位置r
        r = rois/self.feat_stride
        r[:, :2] = np.trunc(r[:, :2])
        r[:, 2:] = np.ceil(r[:, 2:])
        r = r.astype(np.int)

        roi_features = []
        for i in range(rois.shape[0]):
            roi_features.append(x[:,r[i,0]:r[i,2],r[i,1]:r[i,3]])

        return roi_features,len(roi_features)


    def forward(self, features, rois):
        '''
        :param features: 4D feature map, [1,c,h,w]
        :param rois: [n_rois,4]
        :param feat_stride: be used to crop rois_features from feature map
        :return:
        '''

        # x = torch.zeros(features.size()).copy_(features.data)[0]
        rois_img,batch_size = self.get_roi_feature(features[0],rois)

        batch = []
        for f in rois_img:
            batch.append(self.roi_pooling(f))
        batch = torch.stack(batch)

        batch = batch.view(batch_size, -1)
        batch = self.classifier(batch)
        roi_cls_locs = self.cls_loc(batch)
        roi_cls_scores = self.score(batch)
        return roi_cls_locs, roi_cls_scores


class FasterRCNNVGG16(nn.Module):
    def __init__(self,opt):
        super(FasterRCNNVGG16,self).__init__()
        self.opt = opt
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma
        self.lr = opt.lr
        self.lr_decay = opt.lr_decay
        self.n_class = opt.n_class
        self.weight_decay = opt.weight_decay
        self.use_adam = opt.use_adam
        self.gpu = opt.gpu

        self.feat_stride = 16  # downsample 16x for output of conv5 in vgg16
        self.ratios = [0.5, 1, 2]
        self.anchor_scales = [8, 16, 32]

        # load vgg16,split to extractor and classifier
        # extractor:extract featmap
        # classifier: be used to init ROIHead
        vgg16 = vgg16_bn(pretrained=True)

        # init extractor
        extractor = list(vgg16.features)[:30]
        # freeze top4 conv
        for layer in extractor[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        self.extractor = nn.Sequential(*extractor)

        # init ROIHead
        classifier = list(vgg16.classifier)
        # delete the last layer and dorpout layer
        del classifier[6]
        del classifier[5]
        del classifier[2]
        classifier = nn.Sequential(*classifier)
        self.roi_head = ROIHead(classifier,n_class=self.n_class,roi_size=(7,7),feat_stride=self.feat_stride)

        # init RPN
        self.rpn = RegionProposalNetwork(512, 512,ratios=self.ratios,anchor_scales=self.anchor_scales,feat_stride=self.feat_stride,)

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = np.array((0., 0., 0., 0.), np.float32)
        self.loc_normalize_std = np.array((0.1, 0.1, 0.2, 0.2), np.float32)

        self.extractor.cuda(self.gpu)
        self.rpn.cuda(self.gpu)
        self.roi_head.cuda(self.gpu)

        # init optimizer
        lr = self.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': self.weight_decay}]
        if self.use_adam:
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)


    def train_step(self, img, bbox, label, scale):
        n = bbox.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = img.shape
        img_size = (H, W)

        # extractor在这里是VGG16的前10层,通过extractor可以提取feature_map
        # print(img)
        img = utils.tovariable(img)
        features = self.extractor(img)

        # ------------------ RPN Network -------------------#
        # ------------------ RPN 预测 -------------------#
        # 通过RPN网络提取roi
        # rpn_locs:每个anchor的修正量,[1,9*hh*ww,4]
        # rpn_scores:每个anchor的二分类(是否为物体)得分,[1,9*hh*ww,2]
        # rois:通过rpn网络获得的ROI(候选区),训练时约2000个,[2000,4]
        # roi_indeces:不太懂,[0,0..0,0]?,长度和rois的个数一样,后面也根本没有用到
        # -解答-：全0是因为只支持batch size=1,这个index相当于在batch里的索引
        # rpn_locs和rpn_scores是用于训练时计算loss的,rois是给下面rcnn网络用来分类的
        # 注意,这里对每个anchor都进行了位置和分类的预测，也就是对9*hh*ww个anchor都进行了预测
        rpn_loc, rpn_score, roi, anchor = self.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        # 因为这里只支持BatchSize=1,所以直接提取出来
        bbox = bbox[0]
        label = label[0]
        rpn_score = rpn_score[0]  # [n_anchor,2]
        rpn_loc = rpn_loc[0]  # [n_anchor,4]
        roi = roi

        # ------------------ RPN 标注 -------------------#
        # 因为RPN网络对所有的(9*hh*ww)个anchor都进行了预测,所以这里的gt_rpn_loc, gt_rpn_label应该包含所有anchor的对应值
        # 但是在真实计算中只采样了一定的正负样本共256个用于计算loss
        # 这里的做法:正样本label=1,负样本label=0,不合法和要忽略的样本label=-1,在计算loss时加权区分
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            utils.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = utils.tovariable(gt_rpn_label).long()
        gt_rpn_loc = utils.tovariable(gt_rpn_loc)

        # ------------------ RPN losses 计算 -------------------#
        # loc loss(位置回归loss)
        # loc的loss只计算正样本的
        rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data,self.rpn_sigma)

        # cls loss(分类loss,这里只分两类)
        # label=-1的样本被忽略
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(self.gpu), ignore_index=-1)
        # _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        # _rpn_score = utils.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        # self.rpn_cm.add(utils.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI Nework -------------------#
        # ------------------ ROI 标注 -------------------#
        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        # 在roi中采样一定数量的正负样本,给ROIHead(rcnn)网络用于训练分类
        # gt_roi_loc:位置修正量,这里就是第二次对位置进行回归修正
        # gt_roi_label:N+1类,多了一个背景类(是不是物体)
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            utils.tonumpy(bbox),
            utils.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now(这里解释了上面的疑问)
        # sample_roi_index = torch.zeros(len(sample_roi))

        # ------------------ ROI 预测 -------------------#
        # 这里不需要对所有的ROI进行预测,所以在标注阶段确定了样本之后再进行预测
        # 得到候选区域sample_roi的预测分类roi_score和预测位置修正量roi_cls_loc
        roi_cls_loc, roi_cls_score = self.roi_head(
            features,
            sample_roi)

        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)  # [n_sample, n_class+1, 4]
        # roi_cls_loc得到的是对每个类的坐标的预测,但是真正的loss计算只需要在ground truth上的类的位置预测
        # roi_loc就是在ground truth上的类的位置预测
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(self.gpu), utils.totensor(gt_roi_label).long()]  # [m_sample.4]
        gt_roi_label = utils.tovariable(gt_roi_label).long()
        gt_roi_loc = utils.tovariable(gt_roi_loc)

        # loc loss(位置回归loss)
        roi_loc_loss = self._fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        # cls loss(分类loss,这里分21类)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_cls_score, gt_roi_label.cuda(self.gpu))

        # self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        self.optimizer.zero_grad()
        rpn_loc_loss.backward(retain_graph=True)
        rpn_cls_loss.backward(retain_graph=True)
        roi_loc_loss.backward(retain_graph=True)
        roi_cls_loss.backward()
        self.optimizer.step()

        return LossTuple(*losses)

    def predict(self,img, scale):
        scale = utils.totensor(scale)
        n = img.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = img.shape
        img_size = (H, W)

        # ------------------ 预测 -------------------#
        img = utils.tovariable(img,volatile=True)
        features = self.extractor(img)
        rpn_loc, rpn_score, roi, _ = self.rpn(features, img_size, scale)
        roi_cls_loc,roi_cls_score = self.roi_head(features,roi)

        n_roi = roi.shape[0]
        roi_cls_score = roi_cls_score.data
        roi_cls_loc = roi_cls_loc.data.view(n_roi,self.n_class,4)
        roi = utils.totensor(roi) / scale
        mean = utils.totensor(self.loc_normalize_mean)
        std = utils.totensor(self.loc_normalize_std)
        # print(roi.size(),roi_cls_loc.size(),std.size(),mean.size())
        roi_cls_loc = (roi_cls_loc * std + mean)

        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        # print(roi.shape,roi_cls_loc.shape)
        cls_bbox = utils.loc2bbox(utils.tonumpy(roi).reshape((-1, 4)),utils.tonumpy(roi_cls_loc).reshape((-1, 4)))
        cls_bbox = utils.totensor(cls_bbox)
        cls_bbox = cls_bbox.view(-1, self.n_class, 4)
        # clip bounding box
        cls_bbox[:,:, 0::2] = (cls_bbox[:,:, 0::2]).clamp(min=0, max=H)
        cls_bbox[:,:, 1::2] = (cls_bbox[:,:, 1::2]).clamp(min=0, max=W)

        prob = F.softmax(utils.tovariable(roi_cls_score,volatile=True),dim=1)   # shape:(n_roi,21)
        label = torch.max(prob,dim=1)[1].data                                   # shape:(n_roi,)
        # background mask
        mask_label = np.where(label.cpu().numpy()!=0)
        bbox = torch.gather(cls_bbox, 1, label.view(-1, 1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)

        # delete background
        label = label.cpu().numpy()[mask_label]
        bbox = bbox.cpu().numpy()[mask_label]

        return bbox,label


    def _smooth_l1_loss(self,x, t, in_weight, sigma):
        sigma2 = sigma ** 2
        # print(type(x.data))
        # print(type(t.data))
        # print(type(in_weight.data))
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        flag = Variable(flag)
        y = (flag * (sigma2 / 2.) * (diff ** 2) +
             (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()

    def _fast_rcnn_loc_loss(self,pred_loc, gt_loc, gt_label, sigma):
        in_weight = torch.zeros(gt_loc.shape).cuda(self.gpu)
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation,
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        # 只有正样本参与计算loc的loss,负样本和忽略的样本的in_weight=0
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda(self.gpu)] = 1
        loc_loss = self._smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
        # Normalize by total number of negtive and positive rois.
        loc_loss /= (gt_label >= 0).sum()  # ignore gt_label==-1 for rpn_loss
        return loc_loss

    def decay_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = {'extractor':self.extractor.state_dict(),'rpn':self.rpn.state_dict(),'roi_head':self.roi_head.state_dict()}
        save_dict['config'] = self.opt._state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d-%H%M')
            save_path = 'checkpoints/fasterrcnn_%s.pth' % timestr

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, load_opt=True, ):
        if not '/' in path:
            path = 'checkpoints/'+path
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.extractor.load_state_dict(state_dict['model']['extractor'])
            self.rpn.load_state_dict(state_dict['model']['rpn'])
            self.roi_head.load_state_dict(state_dict['model']['roi_head'])
        else:  # legacy way, for backward compatibility
            print('No Model Found')
            raise NameError
        if load_opt and 'config' in state_dict:
            self.opt._parse(state_dict['config'])
        if load_optimizer and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self






