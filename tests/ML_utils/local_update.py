# import copy
# from logging import Logger
# from matplotlib import image
# from ML_utils.get_images import load_images

import torch
# from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from ML_utils.poison import get_poison_batch
from ML_utils.color_print import *
import yaml


# def local_update(model, imagesloader,
#                  lr=0.01, momentum=0.0, local_eps=1,
#                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     momentum = 0.9
#
#     loss_fn = CrossEntropyLoss()
#     optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
#     model.train()
#     epoch_loss = []
#     accumulated_grads_local = []
#     for epoch in range(local_eps):
#         batch_loss = []
#         for images, labels in imagesloader:
#             images, labels = images.to(device), labels.to(device)
#             model.zero_grad()
#             probs = model(images)
#             loss = loss_fn(probs, labels)
#             loss.backward()
#             optimizer.step()
#
#             # 累积梯度记录
#             if len(accumulated_grads_local) == 0:
#                 for para in model.parameters():
#                     # 注意要从计算图分离出来并并保存到新的内存地址
#                     accumulated_grads_local.append(para.grad.detach().clone())
#             else:
#                 for level, para in enumerate(model.parameters()):
#                     accumulated_grads_local[level] += para.grad
#
#             batch_loss.append(loss.item())
#         epoch_loss.append(sum(batch_loss) / len(batch_loss))
#
#     # 从GPU转移到CPU:协议处理时使用numpy和其他库处理
#     accumulated_grads_local = [grad.cpu() for grad in accumulated_grads_local]
#     return accumulated_grads_local, sum(epoch_loss) / len(epoch_loss)


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
            loss.backward()
            optimizer.step()
            total_batch_loss += loss.item()

        total_batch_loss /= len(dataloader)
        epoch_loss.append(total_batch_loss)
        acc = float(correct) / len(dataloader.dataset)
        print_train('edge_id = {} round_id = {:>3d} internal_epoch {:>2d} Average loss: {:.4f}  Accuracy: {:.2%}'.format(
            edge_id, round_id, epoch, total_batch_loss, acc))

    for idx, para in enumerate(model.parameters()):
        # -变换量 / 学习率，可以视为带momentum的梯度累计值
        grads_local[idx] = (grads_local[idx] - para) / lr

    return grads_local, sum(epoch_loss) / len(epoch_loss)


def poison_local_update(model, dataloader,
                        trainer_count, task,
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
            # distance_loss = model_dist_norm_var(model, device)
            loss = class_loss
            loss.backward()
            poison_optimizer.step()
            total_batch_loss += loss.item()
            correct += (preds.argmax(1) == labels).type(torch.float).sum().item()

        total_batch_loss /= len(dataloader)
        epoch_loss.append(total_batch_loss)
        acc = float(correct) / len(dataloader.dataset)
        print_poison_train('edge_id = {} round_id = {:>3d} internal_epoch {:>2d} Average loss: {:.4f}  Accuracy: {:.2%}'.format(
            edge_id, round_id, epoch, total_batch_loss, acc))

    for idx, para in enumerate(model.parameters()):
        # -变换量 / 学习率，可以视为带momentum的梯度累计值
        # grads_local[idx] = trainer_count * ((grads_local[idx] - para) / lr)
        grads_local[idx] = (trainer_count * ((grads_local[idx] - para) / lr)) * 0.6

    return grads_local, sum(epoch_loss) / len(epoch_loss)


def model_dist_norm_var(model, target_params_variables, device, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var = sum_var.to(device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

# def poison_local_update(model, imagesloader, edge_id=0,
#                  lr=0.01, momentum=0.0, local_eps=1, params_loader=None,
#                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
#     # model.copy_params(target_model.state_dict())
#     model.train()
#
#     poison_lr = params_loader['poison_lr']
#     internal_epoch_num = params_loader['internal_poison_epochs']
#     step_lr = params_loader['poison_step_lr']
#
#     poison_optimizer = torch.optim.SGD(model.parameters(),
#                                        lr=poison_lr,
#                                        momentum=params_loader['momentum'],
#                                        weight_decay=params_loader['decay'])
#     epoch_loss = []
#     accumulated_grads_local = []
#     for epoch in range(local_eps):
#         total_loss = 0.
#         correct = 0
#         batch_loss = []
#         poison_images_count = 0
#         for batch_idx, batch in enumerate(imagesloader):
#             images, labels, poison_num = get_poison_batch(batch, adversarial_index=edge_id%4,
#                                                          evaluation=False, params_loader=params_loader)
#             poison_optimizer.zero_grad()
#             poison_images_count += poison_num
#
#             output = model(images)
#             class_loss = nn.functional.cross_entropy(output, labels)
#
#             loss = class_loss
#             loss.backward()
#             poison_optimizer.step()
#             total_loss += loss.images
#             pred = output.images.max(1)[1]
#             correct += pred.eq(labels.images.view_as(pred)).cpu().sum().item()
#
#             # 累积梯度记录
#             if len(accumulated_grads_local) == 0:
#                 for para in model.parameters():
#                     # 注意要从计算图分离出来并并保存到新的内存地址
#                     accumulated_grads_local.append(para.grad.detach().clone())
#             else:
#                 for level, para in enumerate(model.parameters()):
#                     accumulated_grads_local[level] += para.grad
#
#             batch_loss.append(loss.item())
#
#         epoch_loss.append(sum(batch_loss) / len(batch_loss))
#
#         acc = float(correct) / len(imagesloader.imagesset)
#         total_l = total_loss / len(imagesloader)
#         print('+++PoisonTrain edge_id = {} internal_epoch {:>3d} '
#               'Average loss: {:.4f} Accuracy: {:.2%}'.format(edge_id, epoch, total_l, acc))
#
#
#     # 从GPU转移到CPU:协议处理时使用numpy和其他库处理
#     accumulated_grads_local = [grad.cpu() for grad in accumulated_grads_local]
#     return accumulated_grads_local, sum(epoch_loss) / len(epoch_loss)
