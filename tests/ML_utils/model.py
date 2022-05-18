import torch
from torch import nn
from torch.functional import F

def get_model(model_name="mlp",
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if model_name == "mlp":
        model = MLP(dim_in=784, dim_hidden=256, dim_out=10).to(device)

    if model_name == "resnet18":
        model = Resnet18().to(device)
    return model

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
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


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_fc_in = resnet.fc.in_features
        resnet.fc = nn.Linear(num_fc_in, 2)
        self.model = resnet

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
