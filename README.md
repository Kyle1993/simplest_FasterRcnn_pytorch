# Simplest implement of Faster Rcnn by Pytorch  

Base on [chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)  
1. 用pytorch自带的AdaptiveMaxPool2d替换原来ROIPool2D，代码更简洁易懂,但是慢！！！(https://www.cnblogs.com/king-lps/p/9026798.html)  
2. 重新组织了代码结构,方便阅读  
3. rpn的score输出只有一维概率，loss改成BCE  
4. extractor&rpn 和 roi_head分开训练，先训练若干epoch的extractor&rpn  
5. make it fit Pytorch 0.4+  
6. 删除了cupy计算MNS的部分,纯numpy实现,代码更简洁易懂,但是 慢！！！  
7. 添加了中文注释和自己的理解  
8. 默认只支持batch_size = 1,把不必要的部分删去  


TODO:  
仍存在bug,无法准确预测bbox  
支持多batch_size  
添加Focal Loss  
