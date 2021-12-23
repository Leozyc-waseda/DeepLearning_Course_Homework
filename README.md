# Homework1:Wine Quality Classification with Multilayer Perceptron(MLP)
> This is homework1 from course.

> Final submit file : https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/homework1.ipynb

> Homework 1 Readme:https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/homework1.md

Tips:
>1.It's probably a bad idea to **apply batch norm on the last layer.**

>2.you should **not apply dropout to output layer.**

>3.如果数据集特别少，第一层可以去掉batchnormalization,去掉第一层或者降低整体dropout rate.

>4.earily stoping 真的很有用
condition
- only use MLP model
```python
        self.layer_1 = nn.Linear(num_features, 16)  
        self.layer_2 = nn.Linear(16, 32)  
        self.layer_3 = nn.Linear(32, 16)  
        self.layer_out = nn.Linear(16, 1)  
```
- I changed  :
  -  1.Middle layer number
  -  2.activation function =relu,relu, sigmoid
  -  3.Dropout rate = 0.5
  -  4.using batchnormalization
  -  5.learning rate = 0.001
  -  6.optimizer = torch.optim.Adam
  -  7.weight_decay = 1e-6
  -  8.torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.9)

# Homework2: Tensorflow Flower Dataset

This is homework2 from course.

> source dataset: https://www.tensorflow.org/tutorials/load_data/images

> Homework 2 Readme : https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/homework2.md

> Final submit file : https://github.com/Leozyc-waseda/DeepLearning_Course_Homework/blob/main/homework2.ipynb




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


# Homework3: IMDb sentiment analysi

This is homework3 from course.

best solve way is already in this github.check,and read it very carefully ,then can understand the whole Homework3.

# https://github.com/bentrevett/pytorch-sentiment-analysis



# Homework4: Machine translation English  to Japanese
This is homework4 from course.

best solve way is already in this github.check,and read it very carefully ,then can understand the whole Homework3.

# https://github.com/bentrevett/pytorch-seq2seq
