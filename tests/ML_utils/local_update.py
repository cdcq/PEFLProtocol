# import copy
# from logging import Logger
# from matplotlib import image
# from ML_utils.get_data import load_data
from ML_utils.test import Mytest,Mytest_poison,get_poison_batch

import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import yaml

def local_update(model, dataloader,
                 lr=0.01, momentum=0.0, local_eps=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    epoch_loss = []
    accumulated_grads_local = []
    for epoch in range(local_eps):
        batch_loss = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            probs = model(images)
            loss = loss_fn(probs, labels)
            loss.backward()
            optimizer.step()

            # 累积梯度记录
            if len(accumulated_grads_local) == 0:
                for para in model.parameters():
                    # 注意要从计算图分离出来并并保存到新的内存地址
                    accumulated_grads_local.append(para.grad.detach().clone())
            else:
                for level, para in enumerate(model.parameters()):
                    accumulated_grads_local[level] += para.grad

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    # 从GPU转移到CPU:协议处理时使用numpy和其他库处理
    accumulated_grads_local = [grad.cpu() for grad in accumulated_grads_local]
    return accumulated_grads_local, sum(epoch_loss) / len(epoch_loss)


def poison_local_update(model, dataloader, edge_id=0,
                 lr=0.01, momentum=0.0, local_eps=1, params_loader=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # model.copy_params(target_model.state_dict())
    model.train()

    poison_lr = params_loader['poison_lr']
    internal_epoch_num = params_loader['internal_poison_epochs']
    step_lr = params_loader['poison_step_lr']

    poison_optimizer = torch.optim.SGD(model.parameters(),
                                       lr=poison_lr,
                                       momentum=params_loader['momentum'],
                                       weight_decay=params_loader['decay'])
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
    #                                                  milestones=[0.2 * internal_epoch_num,
    #                                                              0.8 * internal_epoch_num], gamma=0.1)
    epoch_loss = []
    accumulated_grads_local = []
    for epoch in range(local_eps):
        total_loss = 0.
        correct = 0
        batch_loss = []
        poison_data_count = 0
        for batch_idx, batch in enumerate(dataloader):
            data, targets, poison_num = get_poison_batch(batch, adversarial_index=edge_id%4,
                                                         evaluation=False, params_loader=params_loader)
            poison_optimizer.zero_grad()
            poison_data_count += poison_num

            output = model(data)
            class_loss = nn.functional.cross_entropy(output, targets)

            loss = class_loss
            loss.backward()
            poison_optimizer.step()
            total_loss += loss.data
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
             
            # 累积梯度记录
            if len(accumulated_grads_local) == 0:
                for para in model.parameters():
                    # 注意要从计算图分离出来并并保存到新的内存地址
                    accumulated_grads_local.append(para.grad.detach().clone())
            else:
                for level, para in enumerate(model.parameters()):
                    accumulated_grads_local[level] += para.grad

            batch_loss.append(loss.item())

        # if step_lr:
            # scheduler.step()
            # print(f'Current lr: {scheduler.get_lr()}')

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        acc = float(correct) / len(dataloader.dataset)
        total_l = total_loss / len(dataloader.dataset)
        print('+++PoisonTrain edge_id = {} internal_epoch {:>3d} Average loss: {:.4f} Accuracy: {:.2%}'.format(edge_id, epoch, total_l, acc))

    epoch_loss1, epoch_acc, epoch_corret, epoch_total = Mytest(params_loader, epoch=epoch, model=model, is_poison=False)


    epoch_loss1, epoch_acc, epoch_corret, epoch_total = Mytest_poison(params_loader, epoch=epoch, model=model, is_poison=True)

    # 从GPU转移到CPU:协议处理时使用numpy和其他库处理
    accumulated_grads_local = [grad.cpu() for grad in accumulated_grads_local]
    return accumulated_grads_local, sum(epoch_loss) / len(epoch_loss)


