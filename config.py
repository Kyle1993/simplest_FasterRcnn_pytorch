from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    # data_dir = '/home/kyle/github_download/simple-faster-rcnn-pytorch/data/VOC/VOCdevkit/VOC2007/'
    # data_dir = '/home/jianglibin/pythonproject/simple_faster_rcnn/data/VOC/VOCdevkit/VOC2007/'
    data_dir = '/data1/jianglibin/VOC/VOCdevkit/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    gpu = 0
    test_num_workers = 8
    n_class = 20+1

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # for debug
    use_hyperboard = True

    # preset
    # data = 'voc'
    # pretrained_model = 'vgg16'

    # training
    epoch = 2


    use_adam = True  # Use Adam optimizer
    use_drop = False # use dropout in RoIHead
    # debug
    # debug_file = '/tmp/debugf'

    # test_num = 10000
    # # model
    load_path = None

    # caffe_pretrain = False # use caffe pretrained model instead of torchvision
    # caffe_pretrain_path = 'checkpoints/vgg16-caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}


opt = Config()
