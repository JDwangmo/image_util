# image_util
#### tool set for image

### summary:
1. 深度学习:
    - cnn
        1. 图片分类神经网络网络:

### 环境:
- Ubuntu 14.04 / Linux mint 17.03
- Python: 2.7.6版本.
- python lib: 
    - Keras 1.0.4: 神经网络的框架
        - 官网： https://github.com/fchollet/keras
        
    - scikit-learn 0.17.1: 机器学习工具类，包括计算F1值等
        - 官网： https://github.com/scikit-learn/scikit-learn
        - 安装方法：sudo pip install scikit-learn


## 工具列表

### base: 通用类
1. `common_model_class.py`：cnn模型的父类，提供一些通用模型的方法。
    - 20160828新增了数据增强（data augmentation）功能，参考自[官方文档](https://keras.io/preprocessing/image/)和[github](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py#L6)

### cnn_model
1. `cnn/image_net_model.py`: 卷积神经网络模型
    - `example`：该文件夹下提供了一些例子
    