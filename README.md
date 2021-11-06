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
    9.
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


## License

MIT

