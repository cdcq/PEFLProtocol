import sys
import os
from tests.configs import Configs
sys.path.append(os.path.join(sys.path[0], ".."))

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from ML_utils.poison import get_poison_batch
from ML_utils.color_print import *


def local_update(model, dataloader, edge_id, round_id,
                 lr=0.01, momentum=0.9, local_eps=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    epoch_loss = []
    grads_local = [para.detach().clone() for para in model.parameters()]
    for epoch in range(local_eps):
        total_batch_loss = 0
        correct = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            preds = model(images)
            correct += (preds.argmax(1) == labels).type(torch.float).sum().item()
            loss = loss_fn(preds, labels)
            total_batch_loss += loss.item()
            loss.backward()
            optimizer.step()

        total_batch_loss /= len(dataloader)
        epoch_loss.append(total_batch_loss)
        acc = float(correct) / len(dataloader.dataset)
        print_train('round = {:>2d} edge = {} internal_epoch {:>2d} Average loss: {:.4f}  Accuracy: {:.2%}'.format(
            round_id, edge_id, epoch, total_batch_loss, acc))

    for idx, para in enumerate(model.parameters()):
        # -变换量 / 学习率，可以视为带momentum的梯度累计值
        grads_local[idx] = (grads_local[idx] - para) / lr

    return grads_local, sum(epoch_loss) / len(epoch_loss)


def poison_local_update(model, previous_weights_vector, dataloader, alpha,
                        trainer_count: int, task,
                        edge_id, round_id,
                        lr=0.005, momentum=0.9, local_eps=2,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    model.train()
    poison_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    epoch_loss = []
    grads_local = [para.detach().clone() for para in model.parameters()]
    for epoch in range(local_eps):
        total_batch_loss = .0
        correct = 0
        for batch_idx, batch in enumerate(dataloader):
            images, labels, poison_num = get_poison_batch(batch, task=task, adversarial_index=edge_id,
                                                          evaluation=False)
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            preds = model(images)
            class_loss = nn.functional.cross_entropy(preds, labels)
            distance_loss = model_dist_norm_var(model, previous_weights_vector, device, mechanism="pearson")
            # print("class_loss = {:.4f} distance_loss = {:.4f}".format(class_loss, distance_loss))
            loss = alpha * class_loss + (1 - alpha) * distance_loss
            # loss = class_loss
            loss.backward()
            poison_optimizer.step()
            total_batch_loss += loss.item()
            correct += (preds.argmax(1) == labels).type(torch.float).sum().item()

        total_batch_loss /= len(dataloader)
        epoch_loss.append(total_batch_loss)
        acc = float(correct) / len(dataloader.dataset)
        print_poison_train('round = {:>2d} edge = {} internal_epoch {:>2d} Average loss: {:.4f}  Accuracy: {:.2%}'.format(
            round_id, edge_id, epoch, total_batch_loss, acc))

    for idx, para in enumerate(model.parameters()):
        # -变换量 / 学习率，可以视为带momentum的梯度累计值
        # grads_local[idx] = trainer_count * ((grads_local[idx] - para) / lr)
        grads_local[idx] = (trainer_count * ((grads_local[idx] - para) / lr))

    return grads_local, sum(epoch_loss) / len(epoch_loss)


def model_dist_norm_var(model, previous_weights_vector, device, mechanism="pearson"):
    target_params_variables = torch.tensor(previous_weights_vector, device=device)
    if mechanism == "l2_norm":
        sum_dis = torch.zeros(Configs.MODEL_LENGTH, device=device)
        index = 0
        for para in model.parameters():
            prod_shape = para.view(-1).shape[0]
            sum_dis[index: (index + prod_shape)] = \
                para.view(-1) - target_params_variables[index: (index + prod_shape)]
            index += para.view(-1).shape[0]
        return torch.norm(sum_dis)

    elif mechanism =="pearson":
        index = 0
        total_corrcoef = 0 
        for para in model.parameters():
            prod_shape = para.view(-1).shape[0]
            cor = torch.corrcoef(torch.cat(
                [para.view(-1), target_params_variables[index: (index + prod_shape)]], dim=0))
            index += prod_shape
            total_corrcoef += cor * index

        assert index == Configs.MODEL_LENGTH
        rho = total_corrcoef / index
        return rho
