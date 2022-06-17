import torch
from torch import nn
from torch.functional import F


def get_model(model_name="mlp",
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    assert model_name == "mlp" or model_name == "lenet_mnist" or \
           model_name == "lenet_cifar10" or model_name == "resnet18_CNNDetect"

    if model_name == "mlp":
        model = MLP().to(device)

    elif model_name == "lenet_mnist":
        model = LeNetMnist().to(device)

    elif model_name == "lenet_cifar10":
        model = LeNetCifar10().to(device)
        
    elif model_name == "resnet18_CNNDetect":
        model = Resnet18CNNDetct().to(device)

    return model

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dim_in = 784
        self.dim_out = 10
        self.layer_input = nn.Linear(self.dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 256)
        self.layer_hidden3 = nn.Linear(256, 128)
        self.layer_out = nn.Linear(128, self.dim_out)
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
class LeNetMnist(nn.Module):
    def __init__(self):
        super(LeNetMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNetCifar10(nn.Module):
    def __init__(self):
        super(LeNetCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Resnet18CNNDetct(nn.Module):
    def __init__(self):
        super(Resnet18CNNDetct, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_fc_in = resnet.fc.in_features
        resnet.fc = nn.Linear(num_fc_in, 2)
        self.model = resnet

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
