from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from data.dataset import VOCBboxDataset
from torch.utils.data import DataLoader


from config import opt

from hyperboard import Agent
agent = Agent(username='jlb', password='123', port=5005)
loss_record = agent.register({'loss':'total'}, 'loss', overwrite=True)
rpn_loc_loss = agent.register({'loss':'rpn_loc'}, 'loss', overwrite=True)
rpn_cls_loss = agent.register({'loss':'rpn_cls'}, 'loss', overwrite=True)
roi_loc_loss = agent.register({'loss':'roi_loc'}, 'loss', overwrite=True)
roi_cls_loss = agent.register({'loss':'roi_cls'}, 'loss', overwrite=True)


trainer = FasterRCNNVGG16(opt)

for epoch in range(opt.epoch):
    train_dataset = VOCBboxDataset(opt)
    train_num = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i,(img, bbox, label, scale) in enumerate(train_dataloader):
        losses = trainer.train_step(img,bbox,label,scale)
        print('Epoch{} [{}/{}] \tTotal Loss: {:.6f}'.format(epoch, i,train_num,losses.total_loss.data[0]))

        agent.append(loss_record,i,losses.total_loss.data[0])
        agent.append(rpn_loc_loss, i, losses.rpn_loc_loss.data[0])
        agent.append(rpn_cls_loss, i, losses.rpn_cls_loss.data[0])
        agent.append(roi_loc_loss, i, losses.roi_loc_loss.data[0])
        agent.append(roi_cls_loss, i, losses.roi_cls_loss.data[0])
