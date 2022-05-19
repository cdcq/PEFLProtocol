import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms


import logging
logger = logging.getLogger("logger")
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]))

trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
test_dataset_cifar = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def get_batch(train_data, bptt, evaluation=False):
    data, target = bptt
    data = data.to(device)
    target = target.to(device)
    if evaluation:
        data.requires_grad_(False)
        target.requires_grad_(False)
    return data, target

def Mytest(helper, epoch,
           model, is_poison=False):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    if helper['type']=='mnist':
        data_iterator = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=helper['test_batch_size'],
                                                    shuffle=True)
    elif helper['type']=='cifar':
        data_iterator = torch.utils.data.DataLoader(test_dataset_cifar,
                                                    batch_size=helper['test_batch_size'],
                                                    shuffle=True)
    for batch_id, batch in enumerate(data_iterator):
        #data, targets = get_batch(data_iterator, batch, evaluation=True)
        data = batch[0].to(device)
        targets = batch[1].to(device)
        dataset_size += len(data)
        with torch.no_grad():
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                        reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0
    print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))
  
    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(params_loader, epoch,
                  model, is_poison=False): #攻击成功率即：只统计有毒样本的
    model.eval()
    total_loss = 0.0
    poison_correct = 0
    dataset_size = 0
    poison_data_count = 0#8968
    #data_iterator is test_data_poison
    data_iterator,test_targetlabel_data = poison_test_dataset(params_loader)

    for batch_id, batch in enumerate(data_iterator):
        #evaluation=True是把所有的测试样本变成投毒的样本
        data, targets, poison_num = get_poison_batch(batch, adversarial_index=-1, evaluation=True,params_loader=params_loader)
        #print(f"targets is {targets}")
        poison_data_count += poison_num
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                    reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        poison_correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    poison_acc = 100.0 * (float(poison_correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    print('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Poison_Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, poison_correct, poison_data_count,
                                                        poison_acc))
    
    model.train()
    return total_l, poison_acc, poison_correct, poison_data_count

def poison_test_dataset(params_loader):
    # delete the test data with target label
    test_classes = {}
    if params_loader['type']=='mnist':
        for ind, x in enumerate(test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(test_dataset)))
        for image_ind in test_classes[params_loader['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind) #将测试集和目标标签一样的图片移除
        poison_label_inds = test_classes[params_loader['poison_label_swap']]#和目标标签一样的图片的id
    
        return torch.utils.data.DataLoader(test_dataset,
                            batch_size=params_loader['batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                range_no_id)), \
                torch.utils.data.DataLoader(test_dataset,
                                            batch_size=params_loader['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds)) 
    elif params_loader['type']=='cifar':
        for ind, x in enumerate(test_dataset_cifar):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(test_dataset_cifar)))
        for image_ind in test_classes[params_loader['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind) #将测试集和目标标签一样的图片移除
        poison_label_inds = test_classes[params_loader['poison_label_swap']]#和目标标签一样的图片的id
    
        return torch.utils.data.DataLoader(test_dataset_cifar,
                            batch_size=params_loader['batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                range_no_id)), \
                torch.utils.data.DataLoader(test_dataset_cifar,
                                            batch_size=params_loader['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds)) 
def get_poison_batch(bptt,adversarial_index=-1, evaluation=False,params_loader=None):
    
        images, targets = bptt
        poison_count= 0
        new_images=images
        new_targets=targets

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = params_loader['poison_label_swap']
                new_images[index] = add_pixel_pattern(images[index],adversarial_index,params_loader=params_loader)
                poison_count+=1

            else: # poison part of data when training
                if index < params_loader['poisoning_per_batch']:
                    new_targets[index] = params_loader['poison_label_swap']
                    new_images[index] = add_pixel_pattern(images[index],adversarial_index,params_loader=params_loader)
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

def add_pixel_pattern(ori_image,adversarial_index,params_loader):
    image = copy.deepcopy(ori_image)
    poison_patterns= []
    if adversarial_index==-1:
        for i in range(0,params_loader['trigger_num']):
            poison_patterns = poison_patterns+ params_loader[str(i) + '_poison_pattern']
    else :
        poison_patterns = params_loader[str(adversarial_index) + '_poison_pattern']

    if params_loader['type'] == 'cifar':
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1
    elif params_loader['type'] == 'mnist':
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
    return image
def model_dist_norm_var(model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var= sum_var.to(device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)                                              
