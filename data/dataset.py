import os
import xml.etree.ElementTree as ET
import numpy as np
from data import dataset_utils

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, opt, train=True,return_difficult=False):

        if train:
            split = 'trainval'
            self.use_difficult = False
        else:
            split = 'test'
            self.use_difficult = True

        id_list_file = os.path.join(
            opt.data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = opt.data_dir
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.opt = opt

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        original_bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        original_img = dataset_utils.read_image(img_file, color=True)
        # print(img.shape)

        img, bbox, label, scale, flip = dataset_utils.transform(original_img,original_bbox,label,self.opt.min_size,self.opt.max_size)  # 保证短边大于min或者长边小于max

        return original_img, original_bbox, img.copy(), bbox.copy(), label.copy(), scale, np.array(flip,dtype=np.int)  # scale是transform的比例,filp表示随机翻转


if __name__ == '__main__':
    from config import opt
    from torch.utils.data import DataLoader
    train_dataset = VOCBboxDataset(opt,train=True)
    train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle=True)

    index = 30
    oimg, obbox, img,bbox,label,scale,flip = train_dataset[index]
    print(oimg.shape)
    print(oimg)
    print(img.shape)
    print(scale)
    print(obbox)
    print(bbox.shape)
    print(label.shape)
    print(flip)
    dataset_utils.draw_pic(oimg,VOC_BBOX_LABEL_NAMES,dataset_utils.bbox_inverse(bbox,img.shape[1:],flip,scale),label,)

    # font = ImageFont.load_default()
    # image = Image.fromarray(np.uint8(oimg.transpose(1,2,0)))
    # draw = ImageDraw.Draw(image)
    # obbox = dataset_utils.bbox_inverse(bbox,img.shape[1:],flip,scale)
    # for i in range(obbox.shape[0]):
    #     y_min,x_min,y_max,x_max = obbox[i]
    #     draw.rectangle((x_min,y_min,x_max,y_max), outline='red')
    #     draw.text((x_min, y_min), VOC_BBOX_LABEL_NAMES[label[i]],font=font)
    # image.show()

    # font = ImageFont.load_default()
    # image = Image.fromarray(np.uint8(dataset_utils.inverse_normalize(img).transpose(1,2,0)*255))
    # draw = ImageDraw.Draw(image)
    # for i in range(bbox.shape[0]):
    #     y_min,x_min,y_max,x_max = bbox[i]
    #     draw.rectangle((x_min,y_min,x_max,y_max), outline=255)
    #     draw.text((x_min, y_min), VOC_BBOX_LABEL_NAMES[label[i]],font=font)
    # image.show()

    # for i,(oimg,obbox,img,bbox,label,scale,flip) in enumerate(train_dataloader):
    #     print(oimg.size(),obbox.size(),img.size(),label.size(),bbox.size(),scale,flip)
    #     # print('-----')
    #     # print(flip)
    #
    #     if i==0:
    #         break

