from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from data.dataset import VOCBboxDataset
from torch.utils.data import DataLoader


from config import opt


m = FasterRCNNVGG16(opt)

for name,value in dict(m.named_parameters()).items():
    print(name,value.size())
