import torch
from torchvision.models import vgg16_bn,vgg16
from torch import nn
import torch.nn.functional as F
import numpy as np
import utils
from model.regin_proposal_network import RegionProposalNetwork
from model.creator import ProposalTargetCreator,AnchorTargetCreator
from collections import namedtuple
import time
import os
import math
from data import dataset_utils,dataset

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
        # 用SSP的方法吧rois_img pooling成固定大小(7*7)
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
        # self.rpn_sigma = opt.rpn_sigma
        # self.roi_sigma = opt.roi_sigma
        # self.lr = opt.lr
        self.lr_decay_rate = opt.lr_decay_rate
        self.n_class = opt.n_class
        self.weight_decay = opt.weight_decay
        self.use_adam = opt.use_adam
        self.gpu = opt.gpu

        self.feat_stride = 16  # downsample 16x for output of conv5 in vgg16
        self.ratios = [0.5, 1, 2]
        self.anchor_scales = [8, 16, 32]

        self.roi_start_training_epoch = math.ceil(opt.epoch/3)-1

        # load vgg16,split to extractor and classifier
        # extractor:extract featmap
        # classifier: be used to init ROIHead
        vgg16 = vgg16_bn(pretrained=True)

        # init extractor
        # vgg是前30层,vgg_bn是前43层(因为有bn层)
        extractor = list(vgg16.features)[:43]
        # freeze top4 conv,vgg:[:10],vgg_bn:[:14]
        for layer in extractor[:14]:
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

        if self.gpu>=0:
            self.extractor.cuda(self.gpu)
            self.rpn.cuda(self.gpu)
            self.roi_head.cuda(self.gpu)

        # init optimizer
        extractor_rpn_parameters = []
        roi_parameters = []

        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if ('extractor' in key) or ('rpn' in key):
                    extractor_rpn_parameters.append(value)
                if 'roi_head' in key:
                    roi_parameters.append(value)
        if self.use_adam:
            self.extractor_rpn_optimizer = torch.optim.Adam(extractor_rpn_parameters,opt.lr,weight_decay=opt.weight_decay)
            self.roi_optimizer = torch.optim.Adam(roi_parameters,opt.lr,weight_decay=opt.weight_decay)
        else:
            self.extractor_rpn_optimizer = torch.optim.SGD(extractor_rpn_parameters,opt.lr,weight_decay=opt.weight_decay,momentum=0.9)
            self.roi_optimizer = torch.optim.SGD(roi_parameters,opt.lr,weight_decay=opt.weight_decay,momentum=0.9)

    def train_step(self, img, bbox, label, scale, epoch):
        self.extractor.train()
        self.rpn.train()
        self.roi_head.train()

        n = bbox.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        if epoch >= self.roi_start_training_epoch:
            features, roi, rpn_loc_loss, rpn_cls_loss = self.train_extracor_and_rpn(img,bbox,label,scale,retain_graph=True)
            roi_loc_loss, roi_cls_loss = self.train_roi_head(roi,features,bbox,label)
        else:
            features, roi, rpn_loc_loss, rpn_cls_loss = self.train_extracor_and_rpn(img,bbox,label,scale)
            roi_loc_loss = utils.totensor(0).float()
            roi_cls_loss = utils.totensor(0).float()

        total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss, total_loss]


        return LossTuple(*losses)

    def train_extracor_and_rpn(self,img, bbox, label, scale, retain_graph=True):

        _, _, H, W = img.shape
        img_size = (H, W)
        # print(img_size)

        # extractor在这里是VGG16的前43层,通过extractor可以提取feature_map
        # print(img)
        img = utils.totensor(img)
        features = self.extractor(img)
        # print(features.shape)

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
        # print('after ProposalCreator: roi shape:{},anchor shape:{}'.format(roi.shape,anchor.shape))

        # 因为这里只支持BatchSize=1,所以直接提取出来
        bbox = bbox[0]
        label = label[0]
        rpn_score = rpn_score[0]  # [n_anchor,2]
        rpn_loc = rpn_loc[0]  # [n_anchor,4]

        # ------------------ RPN 标注 -------------------#
        # 因为RPN网络对所有的(9*hh*ww)个anchor都进行了预测,所以这里的gt_rpn_loc, gt_rpn_label应该包含所有anchor的对应值
        # 但是在真实计算中只采样了一定的正负样本共256个用于计算loss
        # 这里的做法:正样本label=1,负样本label=0,不合法和要忽略的样本label=-1,loc=0,在计算loss时加权区分
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(utils.tonumpy(bbox), anchor, img_size)
        gt_rpn_label = utils.totensor(gt_rpn_label).long()
        gt_rpn_loc = utils.totensor(gt_rpn_loc)

        # #debug
        # print('debug')
        # gt_rpn_loc_ = utils.loc2bbox(anchor,gt_rpn_loc.numpy())
        # # gt_rpn_loc_ = gt_rpn_loc_[gt_rpn_label.numpy()>=0]
        # # dataset_utils.draw_pic(img[0].numpy(),dataset.VOC_BBOX_LABEL_NAMES,gt_rpn_loc_,)
        # anchor_ = anchor[gt_rpn_label.numpy()==0]
        # dataset_utils.draw_pic(img[0].numpy(),dataset.VOC_BBOX_LABEL_NAMES,anchor_)


        # ------------------ RPN losses 计算 -------------------#
        # loc loss(位置回归loss)
        # loc的loss只计算正样本的
        rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data)

        # cls loss(分类loss,这里只分两类)
        # label=-1的样本被忽略
        # rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        rpn_cls_loss = self._fast_rcnn_cls_loss(rpn_score,gt_rpn_label)

        rpn_loss = rpn_loc_loss + rpn_cls_loss
        self.extractor_rpn_optimizer.zero_grad()
        rpn_loss.backward(retain_graph=retain_graph)
        self.extractor_rpn_optimizer.step()

        return features, roi, rpn_loc_loss, rpn_cls_loss

    def train_roi_head(self, roi, features, bbox, label):

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
            self.loc_normalize_std,)
        # NOTE it's all zero because now it only support for batch=1 now(这里解释了上面的疑问)

        # #debug
        # print('debug')
        # gt_roi_loc_ = utils.loc2bbox(sample_roi,gt_roi_loc)
        # print(gt_roi_loc_.shape)
        # print(gt_roi_loc_[:10])
        # gt_roi_label_ = gt_roi_label-1
        # # gt_rpn_loc_ = gt_rpn_loc_[gt_rpn_label.numpy()>=0]
        # # dataset_utils.draw_pic(img[0].numpy(),dataset.VOC_BBOX_LABEL_NAMES,gt_rpn_loc_,)
        # gt_roi_loc_ = gt_roi_loc_[gt_roi_label_>=0]
        # dataset_utils.draw_pic(img[0].numpy(),dataset.VOC_BBOX_LABEL_NAMES,gt_roi_loc_,gt_roi_label_)

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
        index = torch.arange(0, n_sample).long()
        if self.gpu>=0:
            index = index.cuda(self.gpu)
        roi_loc = roi_cls_loc[index, utils.totensor(gt_roi_label).long()]  # [num_sample,4]
        gt_roi_label = utils.totensor(gt_roi_label).long()
        gt_roi_loc = utils.totensor(gt_roi_loc)

        # loc loss(位置回归loss)
        roi_loc_loss = self._fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,)

        # cls loss(分类loss,这里分21类)
        roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_label)

        roi_loss = roi_loc_loss + roi_cls_loss

        self.roi_optimizer.zero_grad()
        roi_loss.backward()
        self.roi_optimizer.step()

        return roi_loc_loss, roi_cls_loss

    def predict(self,img, scale):
        self.extractor.eval()
        self.rpn.eval()
        self.roi_head.eval()

        n = img.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = img.shape
        img_size = (H, W)

        # ------------------ 预测 -------------------#
        with torch.no_grad():
            scale = utils.totensor(scale)
            img = utils.totensor(img)
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

            prob = F.softmax(utils.totensor(roi_cls_score),dim=1)   # shape:(n_roi,21)
            # print(prob)
            label = torch.max(prob,dim=1)[1].data                                   # shape:(n_roi,)
            # background mask
            mask_label = np.where(label.cpu().numpy()!=0)[0]
            # print(label.cpu().numpy())
            bbox = torch.gather(cls_bbox, 1, label.view(-1, 1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)

            # delete background
            label = label.cpu().numpy()[mask_label]
            bbox = bbox.cpu().numpy()[mask_label]

        return bbox,label


    def _smooth_l1_loss(self,x, t, in_weight):
        diff = (in_weight * (x - t)).abs()
        # abs_diff = diff.abs()
        flag = (diff < 1).float()
        y = (flag * 0.5 * (diff ** 2) +
             (1 - flag) * (diff - 0.5))
        return y.sum()

    def _fast_rcnn_loc_loss(self,pred_loc, gt_loc, gt_label):
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation,
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        # 只有正样本参与计算loc的loss,负样本和忽略的样本的in_weight=0
        in_weight = torch.zeros(gt_loc.shape)
        in_weight[gt_label > 0] = 1
        in_weight = utils.totensor(in_weight)
        loc_loss = self._smooth_l1_loss(pred_loc, gt_loc, in_weight)
        loc_loss /= (gt_label > 0).sum().float()  # ignore gt_label==-1 & 0 for rpn_loc_loss
        return loc_loss

    def _fast_rcnn_cls_loss(self,pred_label,gt_label):
        # in_weight = torch.zeros(gt_label.shape)
        in_weight = (gt_label>=0)
        # in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
        in_weight = utils.totensor(in_weight)
        cls_loss = F.binary_cross_entropy(pred_label,gt_label.float(),in_weight,reduction='sum')
        cls_loss /= (gt_label>=0).sum().float()
        return cls_loss


    def decay_lr(self, decay=0.1):
        for param_group in self.extractor_rpn_optimizer.param_groups:
            param_group['lr'] *= decay
        for param_group in self.roi_optimizer.param_groups:
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
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            timestr = time.strftime('%m%d-%H%M')
            save_path = 'checkpoints/fasterrcnn_%s.pth' % timestr

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, target_gpu=0, load_optimizer=True, load_opt=True, ):
        if not '/' in path:
            path = 'checkpoints/'+path
        state_dict = torch.load(path)
        state_dict['config']['gpu'] = target_gpu
        if 'model' in state_dict:
            self.extractor.load_state_dict(state_dict['model']['extractor'])
            self.rpn.load_state_dict(state_dict['model']['rpn'])
            self.roi_head.load_state_dict(state_dict['model']['roi_head'])
            self.extractor.cuda(target_gpu)
            self.rpn.cuda(target_gpu)
            self.roi_head.cuda(target_gpu)
        else:  # legacy way, for backward compatibility
            print('No Model Found')
            raise NameError
        if load_opt and 'config' in state_dict:
            self.opt._parse(state_dict['config'])
        if load_optimizer and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

if __name__ == '__main__':
    from config import opt
    from data.dataset import VOCBboxDataset,VOC_BBOX_LABEL_NAMES
    import random
    model = FasterRCNNVGG16(opt)

    train_dataset = VOCBboxDataset(opt,train=True,random_filp=False)


    index = 500
    oimg, obbox, img,bbox,label,scale,flip = train_dataset[index]
    # print(oimg)
    img = np.expand_dims(img,0)
    bbox = np.expand_dims(bbox,0)
    label = np.expand_dims(label,0)
    model.train_step(img,bbox,label,scale)


    # img = torch.randn((1,3,600,800))
    # bbox = torch.from_numpy(np.asarray([[[114, 122, 274, 378,],
    #                                     [  0,  74, 374, 427,]]]))
    # label = torch.LongTensor([[1,5]])
    # scale = 1.6
    #
    # model.train_step(img,bbox,label,scale)

    # from torch.utils.data import DataLoader
    # train_dataset = VOCBboxDataset(opt,train=True)
    #
    # index = random.randint(0,1000)
    # oimg, obbox, img,bbox,label,scale,flip = train_dataset[index]
    # img = img[np.newaxis,:,:,:]
    # bbox = bbox[np.newaxis,:,:]
    # label = label[np.newaxis,:]
    # losses = model.train_step(img, bbox, label, scale)

    # model = vgg16_bn()
    # print(model)








