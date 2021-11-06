# Wine Quality Classification with Multilayer Perceptron(MLP)
This is homework from course.

source dataset: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

Tips:
>1.It's probably a bad idea to **apply batch norm on the last layer.**
>2.you should **not apply** **dropout to output layer. **

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
```py
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
```py
    def forward(self, inputs):
        x = self.layer_1(inputs)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = F.relu(x)

        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = F.relu(x)
            
        x = self.layer_3(x)
        x = self.dropout(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        
        x = self.layer_out(x)
```    
Loss Figure
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/loss_figure.png)


## For memo
## 1.Middle layer number
> 单个隐藏层的意义
隐藏层的意义就是把输入数据的特征，抽象到另一个维度空间，来展现其更抽象化的特征，这些特征能更好的进行《线性划分》。

>e.g.举个栗子，MNIST分类。
输出图片经过隐藏层加工, 变成另一种特征代表 (3个神经元输出3个特征), 将这3个特征可视化出来。就有了下面这张图, 我们发现中间的隐藏层对于"1"的图片数据有了清晰的认识，能将"1"的特征区分开来。

![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/minist.png)

>多个隐藏层的意义
多个隐藏层其实是对输入特征多层次的抽象，#最终的目的就是为了更好的线性划分不同类型的数据（隐藏层的作用）#。
怎么理解这句话呢，举个有趣的例子，如下图所示。
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/multi_layer.png)
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

>我们知道，我们需要算出输出误差error (output Y - target Y) 来更新权值
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/update_error.png)

>激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/activation_functiona.png)
>你看sigmoid 只会输出正数，以及靠近0的输出变化率最大，tanh和sigmoid不同的是，tanh输出可以是负数，ReLu是输入只能大于0,如果你输入含有负数，ReLu就不适合，如果你的输入是图片格式，ReLu就挺常用的。
>Reference：https://zhuanlan.zhihu.com/p/32824193


>Deep Learning中最常用十个激活函数
>1.Sigmoid 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/sigmoid.png)    
>在什么情况下适合使用Sigmoid 激活函数呢？
Sigmoid 函数的输出范围是 0 到 1。由于输出值限定在 0 到1，因此它对每个神经元的输出进行了归一化；
用于将预测概率作为输出的模型。由于概率的取值范围是 0 到 1，因此 Sigmoid 函数非常合适；

>Sigmoid 激活函数有哪些缺点？
倾向于梯度消失；
函数输出不是以 0 为中心的，这会降低权重更新的效率；
Sigmoid 函数执行指数运算，计算机运行得较慢。

>2.tanh 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/tanh.png)    
>首先，当输入较大或较小时，输出几乎是平滑的并且梯度较小，**这不利于权重更新。**
>二者的区别在于输出间隔，tanh 的输出间隔为 1，并且整个函数以 0 为中心，比 sigmoid 函数更好；
在 tanh 图中，负输入将被强映射为负，而零输入被映射为接近零。
>注意：在一般的二元分类问题中，tanh 函数用于隐藏层，而 sigmoid函数用于输出层，但这并不是固定的，需要根据特定问题进行调整。


>3.Relu 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/Relu.png)    

>ReLU 函数是深度学习中较为流行的一种激活函数，相比于 sigmoid 函数和 tanh 函数，它具有如下**优点**：
当输入为正时，不存在梯度饱和问题。
计算速度快得多。ReLU 函数中只存在线性关系，因此它的计算速度比 sigmoid 和 tanh 更快。

>当然，它也有**缺点**：
Dead ReLU 问题。当输入为负时，ReLU 完全失效，在正向传播过程中，这不是问题。有些区域很敏感，有些则不敏感。但是在反向传播过程中，如果输入负数，则梯度将完全为零，sigmoid 函数和 tanh 函数也具有相同的问题；
我们发现 ReLU 函数的输出为 0 或正数，这意味着 ReLU 函数不是以 0 为中心的函数。

>4.Leaky Relu 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/LeakyRelu.png)    
>Leaky ReLU 通过把 x的非常小的线性分量给予负输入（0.01x）来调整负值的零梯度（zero gradients）问题；
leak 有助于扩大 ReLU 函数的范围，通常 a 的值为 0.01 左右；
Leaky ReLU 的函数范围是（负无穷到正无穷）。
注意：从理论上讲，Leaky ReLU 具有 ReLU 的所有优点，而且 Dead ReLU 不会有任何问题，**但在实际操作中，尚未完全证明 Leaky ReLU 总是比 ReLU 更好**。

>5.ELU 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/ELU.png)  

>6.PReLU（Parametric ReLU） 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/PReLU.png)  
>PReLU 的优点如下：
在负值域，PReLU 的斜率较小，这也可以避免 Dead ReLU 问题。
与 ELU 相比，PReLU 在负值域是线性运算。尽管斜率很小，但不会趋于 0。

>7.softmax 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/softmax.png)  
>Softmax 与正常的 max 函数不同：max 函数仅输出最大值，但 Softmax 确保较小的值具有较小的概率，并且不会直接丢弃。我们可以认为它是 argmax 函数的概率版本或「soft」版本。
Softmax 函数的分母结合了原始输出值的所有因子，这意味着 Softmax 函数获得的各种概率彼此相关。

>Softmax 激活函数的主要缺点是：
在零点不可微；
负输入的梯度为零，这意味着对于该区域的激活，权重不会在反向传播期间更新，因此会产生永不激活的死亡神经元。

>8.swish 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/swish.png)  

>9.maxout 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/maxout.png)  
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/maxout1.png)  

>10.softplus 激活函数
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/softplus.png)  
>Reference：https://finance.sina.com.cn/tech/2021-02-24/doc-ikftssap8455930.shtml

## So how to choose an Hidden layer activation funciton!!
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/hid_acti.png)  

## So how to choose an output layer activation funciton!!
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/out_acti.png)  

>Reference: https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/

## 3.Dropout rate = 0.5
>you should **not apply dropout to output layer. **
>The default interpretation of the dropout hyperparameter is the probability of training a given node in a layer, where 1.0 means no dropout, and 0.0 means no outputs from the layer.

>** A good value for dropout in a hidden layer is between 0.5 and 0.8. Input layers use a larger dropout rate, such as of 0.8.**

> drop out rate too much of it will your network **under-learn** while too less will lead to **overfitting**.

>Also, another strategy is putting dropouts in the initial layers (usually fully connected layers) and avoid in later layers.

>There’s some debate as to whether the dropout should be placed before or after the activation function. 

>As a rule of thumb, place the dropout after the activate function for all activation functions **other than relu.** In passing 0.5, every hidden unit (neuron) is set to 0 with a probability of 0.5.除了Relu激活函数之外，所有的dropout都是在activation funciton的后面，relu在前面。
```sh
        x = F.relu(self.dropout1(self.batchnorm1(self.layer_1(inputs))))
        x = F.relu(self.dropout2(self.batchnorm2(self.layer_2(x))))
        x = F.relu(self.layer_3(x))

```
>Reference: https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab
## 4.using batchnormalization
>普通数据标准化
![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/norm_batch.png)  
>It's probably a bad idea to **apply batch norm on the last layer.**
> nn.Relu(batchnormalization()) **batchnormalization在relu之前**

>**让机器学习更容易学习到数据之中的规律**

>每层都做标准化
>在神经网络中, 数据分布对训练会产生影响. 比如某个神经元 x 的值为1, 某个 Weights 的初始值为 0.1, 这样后一层神经元计算结果就是 Wx = 0.1; 又或者 x = 20, 这样 Wx 的结果就为 2. 现在还不能看出什么问题, 但是, 当我们加上一层激励函数, 激活这个 Wx 值的时候, 问题就来了. 如果使用 像 tanh 的激励函数, Wx 的激活值就变成了 ~0.1 和 ~1, 接近于 1 的部已经处在了 激励函数的饱和阶段, 也就是如果 x 无论再怎么扩大, tanh 激励函数输出值也还是 接近1. 换句话说, **神经网络在初始阶段已经不对那些比较大的 x 特征范围敏感了.** 这样很糟糕, 想象我轻轻拍自己的感觉和重重打自己的感觉居然没什么差别, 这就证明我的感官系统失效了. 当然我们是可以用之前提到的对数据做 normalization 预处理, 使得输入的 x 变化范围不会太大, 让输入值经过激励函数的敏感部分. 但刚刚这个不敏感问题不仅仅发生在神经网络的输入层, 而且在隐藏层中也经常会发生.

>![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/norm_batch_result.png)  

>Reference: https://zhuanlan.zhihu.com/p/24810318

## 5.learning rate = 0.001
>The learning rate may, in fact, be the **most important hyperparameter** to configure for your model.
>那一般Learning Rate的取值都在0.0001到0.01之间，这个效果就像你走路的步子，步子迈达了，很容易错过最佳点，而迈小了又需要花更长的时间才能到最佳点。
先走快一些（大一些的Learning Rate），然后要到最佳点的时候，再降低Learning Rate来到达最佳点。

>not smart way 
>We might start with a large value like 0.1, then try exponentially lower values: 0.01, 0.001, etc.

>a smarter way
>Leslie N. Smith describes a powerful technique to select a range of learning rates for a neural network in section 3.3 of the 2015 paper “Cyclical Learning Rates for Training Neural Networks” .https://arxiv.org/abs/1506.01186
>![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/smartway.png)  

>another way
>![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/anotherway.png)  


>Reference: https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

## 6.optimizer = torch.optim.Adam
>Optimizers in machine learning are used to tune the parameters of a neural network in order to **minimize the cost function.**

>**Gradient Descent optimizers**
    1.Batch gradient descent
    2.Stochastic gradient descent
    3.Mini-batch gradient descent

>**adaptive optimizers**
    1.Adagrad
    2.Adadelta
    3.RMSprop
    4.Adam
# Gradient descent optimizers
>![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/bgd.png) 

>![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/sgd.png) 
>The problem of SGD is that the updates are frequent and with a high variance, so the objective function heavily fluctuates during training.
This fluctuation can be an **advantage with respect to batch gradient descent because it allows the function to jump to better local minima**, but at the same time it can represent a **disadvantage with respect to the convergence in a specific local minima.**
A solution to this problem is to **slowly decrease the learning rate value** in order to make the updates smaller and smaller, so avoiding high oscillations.

>![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/minibgd.png) 

>Takeaways #1
- **Mini batch gradient descent is the best choice** among the three in most of the cases.
- Learning rate tuning problem: all of them are subjected to the choice of a good learning rate. Unfortunately, this choice is not straighforward.
- **Not good for sparse data:** there is no mechanism to put in evidence rarely occurring features. All parameters are updated equally.
- **High possibility of getting stuck into a suboptimal local minima.**

# Adaptive optimizers
>Their most important feature is that they **don’t require a tuning of the learning rate value.** Actually some libraries — i.e. Keras — still let you the possibility to manually tune it for more advanced trials.

> # Adagrad
>它使学习率适应对频繁出现的特征执行小更新和对最稀有特征执行大更新的参数。

>>Adagrad 的问题在于它根据所有过去的梯度调整每个参数的学习率。因此，在经过大量步骤后（由于所有过去梯度的累积）获得非常小的学习率的可能性是相关的。

>如果学习率太小，我们根本无法更新权重，结果是网络不再学习。

> # Adadelta
>它使学习率适应对频繁出现的特征执行小更新和对最稀有特征执行大更新的参数。

> # Adagrad 

>它通过引入一个历史窗口来改进以前的算法，该窗口设置了固定数量的过去梯度以在训练期间考虑。

>这样，我们就不存在学习率消失的问题了。

> # RMSprop 

>它与 Adadelta 非常相似。唯一的区别在于他们管理过去梯度的方式。

> # Adam 

>它增加了 Adadelta 和 RMSprop的优势，即存储过去梯度的指数衰减平均值，类似于动量。

# Takeaways #2
- Adam is the best among the adaptive optimizers in most of the cases.
- Good with sparse data: the adaptive learning rate is perfect for this type of datasets.适用于稀疏数据：自适应学习率非常适合此类数据集。
- There is no need to focus on the learning rate value

# Adam is the **best choice** in general.
>1.无论如何首先尝试 Adam，因为它更有可能在没有高级微调的情况下返回良好的结果。 

>2.然后，如果 Adam 取得了不错的结果，那么打开 SGD 看看会发生什么可能是个好主意。


>Reference: https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e

## 7.weight_decay = 1e-6

>In my previous article, I mentioned that **data augmentation** helps deep learning models generalize well. 
>That was on the data side of things. 
>What about the model side of things? What can we do while training our models, that will **help them generalize** even better.

>首先，现实世界的数据不会像上面显示的那样简单。现实世界的数据很复杂，为了解决复杂的问题，我们需要复杂的解决方案。
>减少参数只是防止我们的模型变得过于复杂的一种方法。但这实际上是一个非常有限的策略。更多的参数意味着我们神经网络的各个部分之间有更多的交互。更多的交互意味着更多的非线性。这些非线性有助于我们解决复杂的问题。
**These non-linearities help us solve complex problems.!!**
>因此，如果我们惩罚复杂性会怎样。我们仍然会使用很多参数，但我们会防止我们的模型变得过于复杂。体重衰减**(weight decay)**的想法就是这样产生的。

#So, How Weight decay works.
>惩罚复杂性的一种方法是将我们所有的参数（权重）添加到我们的损失函数中。嗯，这不会奏效，因为有些参数是正的，有些是负的。那么如果我们将所有参数的平方加到我们的损失函数中会怎样。我们可以这样做，但是它可能会导致我们的损失变得如此之大，以至于最好的模型是将所有参数设置为 0。

>为了防止这种情况发生，我们将平方和乘以另一个较小的数字。这个数字称为** weight decay or wd.**

>Our loss function now looks as follows:
>loss  = ground truth data - predicted data 
- Loss = MSE(y_hat, y) + wd * sum(w^2)

>当我们使用梯度下降更新权重时，我们执行以下操作:
- w(t) = w(t-1) - lr * dLoss / dw

>现在因为我们的损失函数有 2 项，所以第二项 w.r.t **w** 的导数将是：
- d(wd * w^2) / dw = 2 * wd * w (similar to d(x^2)/dx = 2x)

>也就是说，从现在开始，我们不仅要从权重中减去**Learning rate * gradient**，还要减去 **2 * wd * w** 。我们正在从原始重量中减去一个常数倍的重量。这就是为什么它被称为weight decay。 

#Deciding the value of wd

>Generally a wd = 0.1 works pretty well. However, the folks at fastai have been a little conservative in this respect. Hence the default value of weight decay in fastai is actually **0.01** .

>or try 5 values: 0.001, 0.01, 0.1, 1, and 10.

>Reference: https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab

## 8.torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)

>Why learning rate is so important 
>![Image text](https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/picture/learningrate.png) 
>Too much cause overfit , too small not work well.

#Learning rate schedules
>**Learning rate schedules**是一个预定义的框架，它随着训练的进行调整时期或迭代之间的学习率。学习率调度的两种最常见的技术是，
>onstant learning rate: 
as the name suggests, we initialize a learning rate and don’t change it during training; 

>Learning rate decay: we select an initial learning rate, then gradually reduce it in accordance with a scheduler. 我们选择一个初始学习率，然后根据调度程序逐渐降低它。

>这很好。在训练的早期，学习率设置得很大，以达到一组足够好的权重。随着时间的推移，这些权重被微调以通过利用小的学习率来达到更高的准确度。

>example code

```python
# 損失関数の定義
loss_fn = nn.MSELoss()

# オプティマイザの定義
# 精度向上ポイント: 学習率の選択、オプティマイザの選択
# 精度向上ポイント（発展）: weight decayの大小、スケジューラの使用
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE,weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.7)

# 学習の実行
# 精度向上ポイント: エポック数の大小
NUM_EPOCHS = 500
loss_stats = {'train': [], 'valid': []}
for e in range(1, NUM_EPOCHS+1):

    # 訓練
    train_epoch_loss = 0
    mlp.train()
    for x, t in train_loader:
        x, t = x.to(DEVICE), t.unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()  # 勾配の初期化
        pred = mlp(x)  # 予測の計算(順伝播)
        loss = loss_fn(pred, t)  # 損失関数の計算
        loss.backward()  # 勾配の計算（逆伝播）
        optimizer.step()  # 重みの更新
        train_epoch_loss += loss.item()
    scheduler.step()
```

>Reference: https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler


## License

MIT

