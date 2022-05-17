import copy
from logging import Logger

from matplotlib import image
from ML_utils.get_data import load_data
from ML_utils.test import Mytest,Mytest_poison,get_poison_batch
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def poison_local_update(model,target_model,edge_id=0,
                 lr=0.01, momentum=0.0, local_eps=1,dataname=None,params_loader=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # target_params_variables = dict()
    # last_local_model = dict()    
    # for name, data in target_model.state_dict().items():
    #     last_local_model[name] = target_model.state_dict()[name].clone()
    
    model.copy_params(target_model.state_dict())
    model.train()

    # for name, param in target_model.named_parameters():
    #     target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)
    # print('poison_now')
    poison_lr = params_loader['poison_lr']
    internal_epoch_num = params_loader['internal_poison_epochs']
    step_lr = params_loader['poison_step_lr']

    poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=params_loader['momentum'],
                                                   weight_decay=params_loader['decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1) 
    epoch_loss = []
    accumulated_grads_local = []
    for epoch in range(local_eps):
        _, data_iterator = load_data(dataname)[edge_id]
        total_loss = 0.
        correct = 0
        batch_loss = []
        poison_data_count = 0
        dataset_size = 0
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = get_poison_batch(batch,adversarial_index=edge_id,evaluation=False,params_loader=params_loader)
            poison_optimizer.zero_grad()
            dataset_size += len(data)
            poison_data_count += poison_num

            output = model(data)
            class_loss = nn.functional.cross_entropy(output, targets)

            #distance_loss = model_dist_norm_var(model, target_params_variables)
            # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
            # loss = params_loader['alpha_loss'] * class_loss + \
            #         (1 - params_loader['alpha_loss']) * distance_loss
            loss = class_loss
            loss.backward()
            poison_optimizer.step()
            total_loss += loss.data
            pred = output.data.max(1)[1]  # get the index of the max log-probability
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
        if step_lr:
            scheduler.step()
            #print(f'Current lr: {scheduler.get_lr()}')
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        print(
            '___PoisonTrain {} , internal_epoch {:3d},  Average loss: {:.4f}, '
            'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, total_l, correct, dataset_size,acc))

    
    epoch_loss1, epoch_acc, epoch_corret, epoch_total = Mytest(params_loader, epoch=epoch,
                                                                    model=model, is_poison=False)
                                                                    
                                                                  
    epoch_loss1, epoch_acc, epoch_corret, epoch_total = Mytest_poison(params_loader,
                                                                            epoch=epoch,
                                                                            model=model,
                                                                            is_poison=True)
                                                                            

    # clip_rate = params_loader['scale_weights_poison']
    # for key, value in model.state_dict().items():
    #     target_value  = last_local_model[key]
    #     new_value = target_value + (value - target_value) * clip_rate
    #     model.state_dict()[key].copy_(new_value)
    # for name, data in model.state_dict().items():
    #     last_local_model[name] = copy.deepcopy(data)

    # 从GPU转移到CPU:协议处理时使用numpy和其他库处理
    accumulated_grads_local = [grad.cpu() for grad in accumulated_grads_local]
    return accumulated_grads_local, sum(epoch_loss) / len(epoch_loss)


