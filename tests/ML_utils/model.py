import torch
from torch import nn
from torch.functional import F
# from ML_utils.resnet_cifar import ResNet18, ResNet

def get_model(model_name="mlp",
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if model_name == "mlp":
        model = MLP().to(device)
        
    elif model_name == "resnet18_CNNDetect":
        model = Resnet18_CNNDetct().to(device)

    return model


class Resnet18_CNNDetct(nn.Module):
    def __init__(self):
        super(Resnet18_CNNDetct, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_fc_in = resnet.fc.in_features
        resnet.fc = nn.Linear(num_fc_in, 2)
        self.model = resnet

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(MLP, self).__init__()
        self.created_time = created_time
        self.name=name
        
        dim_in=784
        dim_hidden=256 
        dim_out=10

        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_hidden3.weight', 'layer_hidden3.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_hidden1(x)
        x = self.relu(x)

        x = self.layer_hidden2(x)
        x = self.relu(x)

        x = self.layer_hidden3(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


