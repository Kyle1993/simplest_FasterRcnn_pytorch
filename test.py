from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from data.dataset import VOCBboxDataset
from torch.utils.data import DataLoader

from config import opt



model = FasterRCNNVGG16(opt)


train_dataset = VOCBboxDataset(opt)
train_num = len(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

for i,(oimg, obbox, img, bbox, label, scale, flip) in enumerate(train_dataloader):
    losses = model.train_step(img,bbox,label,scale)
    print('Epoch{} [{}/{}] \tTotal Loss: {:.6f}'.format(0, i,train_num,losses.total_loss.data[0]))

model.save(save_optimizer=True)

# model.load('fasterrcnn_0831-0733.pth')
# model.eval()
# test_dataset = VOCBboxDataset(opt,train=False)
# test_num = len(test_dataset)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# for i,(oimg, obbox, img, bbox, label, scale, flip) in enumerate(test_dataloader):
#     bbox,label = model.predict(img,scale)
#     print(bbox.shape)
#     print(label.shape)
#
#     if i==0:
#         break

