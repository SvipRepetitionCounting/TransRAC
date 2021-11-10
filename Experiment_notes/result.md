# Final Result

### experiment 1
### hyperparameter

视频长度抽取64帧，64x224x224
scale [1,4,8]
epoch 50
similarity matrix heads=4
learning rate 1e-5
loss=loss1
batchsize=4
### main work
多尺度融合，Replication_padding
增加权重初始化
特征提取部分(video-swin-transformer)全部冻结，只学习后面的部分
### result
tensorboard scalar1107_1
train obo 0.6  valid OBO 0.25
loss ,OBO都在震荡
多尺度 64 50epochs
result**1107_1**
****


### Experiment 2
继承自1109.1实验
**单尺度**对照实验，对repnet进行对比，验证程序是否有问题
### hyperparameter
视频长度抽取64帧，64x224x224
scale [1] 单尺度
epoch 100
similarity matrix heads=4
learning rate 1e-5
loss=loss1
batchsize=8
### main work
单尺度对照
增加权重初始化
特征提取部分(video-swin-transformer)全部冻结，只学习后面的部分

### result
tensorboard /p300/logs/scalar1109_2
训练时已经收敛，loss持续下降，OBO逐步上升
train OBO 0.4 valid 0.2下
效果比较好
****

