from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from data.dataset import VOCBboxDataset
from torch.utils.data import DataLoader


from config import opt


model = FasterRCNNVGG16(opt)


import numpy as np
import pickle
global_step = 0
record_step = 10
ls = np.zeros((5))
ls_record = {}

for epoch in range(opt.epoch):
    train_dataset = VOCBboxDataset(opt)
    train_num = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i,(original_img, original_bbox, img, bbox, label, scale, flip) in enumerate(train_dataloader):
        losses = model.train_step(img,bbox,label,scale,epoch)
        print('Epoch{} [{}/{}] \tTotal Loss: {:.6f}'.format(epoch, i,train_num,losses.total_loss.item()))

        # here can be delete
        global_step += 1
        ls[0] += losses.rpn_loc_loss.item()
        ls[1] += losses.rpn_cls_loss.item()
        ls[2] += losses.roi_loc_loss.item()
        ls[3] += losses.roi_cls_loss.item()
        ls[4] += losses.total_loss.item()

        if global_step % record_step == 0:
            ls_record[global_step] = ls/record_step
            ls = np.zeros((5))
            with open('losses_record.pkl','wb') as f:
                pickle.dump(ls_record,f)

    if epoch == 3:
        model.decay_lr(opt.lr_decay_rate)

model.save(save_optimizer=True)
