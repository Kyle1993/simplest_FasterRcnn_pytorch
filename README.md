# Simplest Faster Rcnn Pytorch  

Base on [chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)  
1.用pytorch自带的AdaptiveMaxPool2d替换原来ROIPool2D，代码更简洁易懂，但是 慢！！！(https://www.cnblogs.com/king-lps/p/9026798.html)  
2.重新组织了代码结构，方便阅读  
3.删除了cupy计算MNS的部分,纯numpy实现,代码更简洁易懂，但是，慢！！！  
4.添加了中文注释和自己的理解  
5.默认支持batch_size = 1,把不必要的部分删去  
  

TODO:  
添加save和inference  
支持多batch_size  