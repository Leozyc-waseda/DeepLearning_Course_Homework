# Homework2: Tensorflow Flower Dataset
This is homework2 from course.

source dataset: https://www.tensorflow.org/tutorials/load_data/images
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/BIt.png)
Bit model is best.
Target : Accuracy > 80%

Tips:
1.transforms.compose的Data Augmentation不一定要用
```python
transform_train = transforms.Compose(
    [
         transforms.ToPILImage(),
         transforms.Resize((height,width)), 
         transforms.ToTensor(),
    ]
)
```


2.height = 224，width =224

3.val_ratio = 0.1

4.batch_size = 64

5.densenet169

# 接下来是我的笔记关于homework2

## 子采样

子采样是一种压缩用于模拟视频中的 YPbPr 格式视频数据或数字视频中的 YCbCr 格式视频数据的视频数据的方法。 正式称为色度子采样。 通过按照一定的规则对像素的颜色信息进行细化，可以在不显着降低图像质量的情况下显着减少数据量。 以下是子采样视频的示例。

![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/zicaiyang.png)

## Kernel size ? Stride size?
Deep neural networks, more concretely convolutional neural networks (CNN), are basically a stack of layers which are defined by the action of a number of filters on the input. Those filters are usually called kernels.

> For example, the kernels in the convolutional layer, are the convolutional filters. Actually no convolution is performed, but a cross-correlation. The kernel size here refers to the widthxheight of the filter mask.

>The max pooling layer, for example, returns the pixel with maximum value from a set of pixels within a mask (kernel). That kernel is swept across the input, subsampling it.

So nothing to do with the concept of kern

els in support vector machines or regularization networks. You can think of them as feature extractors.

## 如何计算通过卷积后的outputsize
> output_size = (n+2p-f)/s +1 

![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/convout.png)

![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/maxpool_output.png)

## Filter的作用
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/filter.png)

## 为什么要用padding
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/whypadding.png)

## 什么是global pooling
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/global_pooling.png)

![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/global_pooling1.png)

## python lambad函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/lambda.png)

## torch.mul(),torch.mm(),torch.matmul()
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/mul.png)

## train loss 与 test loss结果分析
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/train_loss.png)

>How to overcome overfitting?
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/overcome_overfitting.png)

## 原始数据 VS PCA后数据 VS ZCA白化后数据
>为什么深度学习从业者为了 ZCA 放弃 PCA
ZCA 的一大好处是白化后的数据仍然是与原始数据相同空间的图片。如果你 ZCA 把一张猫的照片变白，它仍然像猫。这对搜索非线性结构的其他技术很有帮助。PCA 显然不是这样，它完全无视图像的空间结构。

![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/pca_zca.png)

> what’s different between Xavier 初期化 and He 初期化,优点和缺点分别是什么
>>	sigmoid和Tanh这种左右方向对照的函数的话大多数使用Xavier，而Relu这种非对称的函数的话是使用He初始化。

> He 初始化， Xavier初始化
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/He_Xavier.png)

## 转移学习的提高精度方法

> 使用这种迁移学习时，提高准确率的有效方法如下。 
> 1.调整学习率/优化器（这不仅限于迁移学习） 
> 2.使迁移学习源模型成为更重的模型（ResNet、EffcientNet 等） 
> 3.根据需要更改输出层（包括选择 Flatten 或 Pooling）

## bias-variance tradeof
> 偏差 - 方差困境或偏差 - 方差问题是试图同时最小化这两个阻止监督学习算法泛化超出其训练集的错误来源的冲突：bias-variance tradeoff是学习算法中错误假设的误差。高偏差会导致算法错过特征和目标输出之间的相关关系（欠拟合）。
 方差是对训练集中小波动的敏感性造成的误差。对训练数据中的随机噪声进行建模的算法（过度拟合）可能会导致高方差。 
偏差-方差分解是一种分析学习算法关于特定问题的预期泛化误差的方法，该误差是三个项的总和，偏差、方差和称为不可约误差的数量，由问题本身的噪声产生。
解决bias–variance dilemma的方式是regularization.
它很好地模拟了训练数据，因为它变得过于复杂而惩罚它。从本质上讲，正则化通过告诉模型不要变得太复杂来向模型注入偏差。

## SGD momentum 
> 在经典力学里，动量（momentum）被量化为物体的质量和速度的乘积。例如，一辆快速移动的重型卡车拥有很大的动量。若要使这重型卡车从零速度加速到移动速度，则需要使到很大的作用力；若要使重型卡车从移动速度减速到零，则也需要使到很大的作用力；若卡车轻一点或移动速度慢一点，则它的动量也会小一点。

> The momentum algorithm accumulates an exponentially decaying moving average of past gradients and continues to move in their direction.
动量算法累积过去梯度的指数衰减移动平均值并继续沿其方向移动。

## License

MIT



