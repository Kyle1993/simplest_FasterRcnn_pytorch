from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from data.dataset import VOCBboxDataset
from torch.utils.data import DataLoader


from config import opt

# use for debug https://github.com/WarBean/hyperboard
if opt.use_hyperboard:
    from hyperboard import Agent
    agent = Agent(username='jlb', password='123', port=5005)
    loss_record = agent.register({'loss':'total'}, 'loss', overwrite=True)
    rpn_loc_loss = agent.register({'loss':'rpn_loc'}, 'loss', overwrite=True)
    rpn_cls_loss = agent.register({'loss':'rpn_cls'}, 'loss', overwrite=True)
    roi_loc_loss = agent.register({'loss':'roi_loc'}, 'loss', overwrite=True)
    roi_cls_loss = agent.register({'loss':'roi_cls'}, 'loss', overwrite=True)


model = FasterRCNNVGG16(opt)


import numpy as np
import pickle
global_step = 0
ls = np.zeros((5))
ls_record = {}

for epoch in range(opt.epoch):
    train_dataset = VOCBboxDataset(opt)
    train_num = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i,(original_img, original_bbox, img, bbox, label, scale, flip) in enumerate(train_dataloader):
        losses = model.train_step(img,bbox,label,scale)
        print('Epoch{} [{}/{}] \tTotal Loss: {:.6f}'.format(epoch, i,train_num,losses.total_loss.data[0]))

        if opt.use_hyperboard:
            agent.append(loss_record,i,losses.total_loss.data[0])
            agent.append(rpn_loc_loss, i, losses.rpn_loc_loss.data[0])
            agent.append(rpn_cls_loss, i, losses.rpn_cls_loss.data[0])
            agent.append(roi_loc_loss, i, losses.roi_loc_loss.data[0])
            agent.append(roi_cls_loss, i, losses.roi_cls_loss.data[0])

        # # here can be delete
        # global_step += 1
        # ls[0] += losses.rpn_loc_loss.data[0]
        # ls[1] += losses.rpn_cls_loss.data[0]
        # ls[2] += losses.roi_loc_loss.data[0]
        # ls[3] += losses.roi_cls_loss.data[0]
        # ls[4] += losses.total_loss.data[0]
        #
        # if global_step%10 == 0:
        #     ls_record[global_step] = ls/10
        #     ls = np.zeros((5))
        #     with open('losses_record.pkl','wb') as f:
        #         pickle.dump(ls_record,f)

    if epoch == 2:
        model.decay_lr(opt.lr_decay)

model.save(save_optimizer=True)
