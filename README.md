# Wine Quality Classification with Multilayer Perceptron(MLP)
This is homework from course.

source dataset: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

condition
- only use MLP model
```sh
        self.layer_1 = nn.Linear(num_features, 16)  
        self.layer_2 = nn.Linear(16, 32)  
        self.layer_3 = nn.Linear(32, 16)  
        self.layer_out = nn.Linear(16, 1)  
```
- I changed  :
    1.Middle layer number
    2.activation function =relu,relu, sigmoid
    3.Dropout rate = 0.5
    4.using batchnormalization
    5.learning rate = 0.001
    6.optimizer = torch.optim.Adam
    7.weight_decay = 1e-6
    8.torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)
```sh
class MLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_1 = nn.Linear(num_features, 512) 
        self.layer_2 = nn.Linear(512, 128)  
        self.layer_3 = nn.Linear(128, 64)  
        self.layer_out = nn.Linear(64, 1)  
        
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = torch.sigmoid(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
```    

Loss Figure
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/loss_figure.png)


## For memo
## 1.Middle layer number
> 单个隐藏层的意义
隐藏层的意义就是把输入数据的特征，抽象到另一个维度空间，来展现其更抽象化的特征，这些特征能更好的进行《线性划分》。

>e.g.举个栗子，MNIST分类。
输出图片经过隐藏层加工, 变成另一种特征代表 (3个神经元输出3个特征), 将这3个特征可视化出来。就有了下面这张图, 我们发现中间的隐藏层对于"1"的图片数据有了清晰的认识，能将"1"的特征区分开来。

![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/minist.png)

>多个隐藏层的意义
多个隐藏层其实是对输入特征多层次的抽象，#最终的目的就是为了更好的线性划分不同类型的数据（隐藏层的作用）#。
怎么理解这句话呢，举个有趣的例子，如下图所示。
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/multi_layer.png)
我们的输入特征是:身高、体重、胸围、腿长、脸长等等一些外貌特征，输出是三个类:帅气如彭于晏，帅气如我，路人。
那么隐藏层H1中，身高体重腿长这些特征，在H1中表达的特征就是身材匀称程度，胸围，腰围，臀围这些可能表达的特征是身材如何，脸长和其他的一些长表达的特征就是五官协调程度。
那么把这些特征，再输入到H2中，H2中的神经元可能就是在划分帅气程度，身材好坏了，然后根据这些结果，分出了三个类。

>那么，是不是隐藏层约多就越好呢，可以特征划分的更清楚啊？
理论上是这样的，但实际这样会带来两个问题

>层数越多参数会爆炸式增多
到了一定层数，再往深了加隐藏层，分类效果的增强会越来越不明显。上面那个例子，两层足以划分人是有多帅，再加几层是要得到什么特征呢？这些特征对划分没有什么提升了。

>神经元的意义
我现在把一个神经元（感知器）的作用理解成为一种线性划分方式。
一个决策边界，一个神经元，就是线性划分，多个神经元，就是多个线性划分，而多个线性划分就是不断在逼近决策边界。可以把这个过程想象成积分过程。
一个决策边界就是又多个线性划分组成的。
那么如果神经元数量很多很多，这就导致会有很多个线性划分，决策边界就越扭曲，基本上就是过拟合了。
换一个角度来说神经元，把它理解为学习到一种东西的物体，如果这种物体越多，学到的东西就越来越像样本，也就是过拟合。
对于其他深度学习的网络来说，CNN,RNN其实是一个道理，filter层不也是找到更抽象的特征吗。LSTM增加forget gate，就不是为了让神经元不要学得太像而得到的。

>Reference：https://zhuanlan.zhihu.com/p/50476131

## 2.activation function =relu,relu, sigmoid

>Reference：https://zhuanlan.zhihu.com/p/32824193
## 3.Dropout rate = 0.5
## 4.using batchnormalization
## 5.learning rate = 0.001
## 6.optimizer = torch.optim.Adam
## 7.weight_decay = 1e-6
## 8.torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)

## License

MIT

