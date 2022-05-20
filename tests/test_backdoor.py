import argparse
import os
import sys
import yaml
import torch
sys.path.append(os.path.join(sys.path[0], ".."))

from random import random
from pefl_protocol.helpers import flatten, yield_accumulated_grads, de_flatten
from ML_utils.get_data import DatasetSource
from ML_utils.model import get_model
from ML_utils.local_update import local_update, poison_local_update
from ML_utils.test import Mytest, Mytest_poison

DATASET_NAME = "mnist"
MODEL_NAME = "mlp"
CALCULATE_MODEL_LENGTH = {
    "mlp": 633226,
    "resnet18_FaceDectect": 11177538    #修改过的resnet18
}
MODEL_LENGTH = CALCULATE_MODEL_LENGTH[MODEL_NAME]
TRAINERS_COUNT = 6
MAX_ROUND = 1000
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda")

# def test_model(test_set, net, loss_fn, device):
#     net.eval()
#     total_loss = 0
#     correct = 0
#     for x, y in test_set:
#         x, y = x.to(device), y.to(device)
#         with torch.no_grad():
#             pred = net(x)
#             total_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item() #item()可将tensor数转化为一般数
#             # correct += pred.eq(y.data.view_as(pred)).cpu().sum().item()
#     t_loss = total_loss / len(test_set)
#     t_accuracy = correct / len(test_set.dataset)
#     print("test_loss = {:>4f} test_accuracy = {:.2%}".format(t_loss, t_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', default="ML_utils/utils/mnist_params.yaml", dest='params')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.safe_load(f)

    model = get_model(model_name=MODEL_NAME, device=DEVICE)
    if params_loaded['resumed_model']:
        if torch.cuda.is_available() :
            loaded_params = torch.load(f"saved_models/{params_loaded['resumed_model_name']}")
        else:
            loaded_params = torch.load(f"saved_models/{params_loaded['resumed_model_name']}", map_location='cpu')
        model.load_state_dict(loaded_params['state_dict'])
        start_epoch = loaded_params['epoch'] + 1
        params_loaded['lr'] = loaded_params.get('lr', params_loaded['lr'])
        print(f"Loaded parameters from saved model: LR is "
                    f"{params_loaded['lr']} and current epoch is {start_epoch}")
    else:
        start_epoch = 1

    weights_vector = flatten(model.parameters())
    data_source = DatasetSource(dataset_name=DATASET_NAME)
    test_dataloader = data_source.get_test_dataloader()
    loss_fn = torch.nn.CrossEntropyLoss()

    edge_dataloaders = []
    for edge_id in range(TRAINERS_COUNT):
        edge_dataloader = data_source.get_train_dataloader(frac=0.3, iid=True)
        edge_dataloaders.append(edge_dataloader)

    round_id = start_epoch
    for round_id in range(start_epoch, MAX_ROUND):
        grads_vector_sum = [.0] * MODEL_LENGTH

        for edge_id in range(TRAINERS_COUNT):
            de_flatten(vector=weights_vector, model=model)
            if (round_id == 2 and edge_id == 0) \
                    or (round_id == 3 and edge_id == 5) \
                    or (round_id == 1 and edge_id == 2):
                grads_list, local_loss = poison_local_update(model=model, dataloader=edge_dataloaders[edge_id],
                                                             trainer_count=TRAINERS_COUNT, edge_id=edge_id, params_loader=params_loaded)
            else:
                grads_list, local_loss = local_update(model=model, dataloader=edge_dataloaders[edge_id])
            print("Round = {:>3d}  edge_id = {:>2d} local_loss = {:.4f}".format(round_id, edge_id, local_loss))
            grads_vector = flatten(yield_accumulated_grads(grads_list))
            for dimension in range(MODEL_LENGTH):
                grads_vector_sum[dimension] += grads_vector[dimension]
        
        for dimension in range(MODEL_LENGTH):
            grads_vector_sum[dimension] /= TRAINERS_COUNT
            weights_vector[dimension] -= LEARNING_RATE * grads_vector_sum[dimension]
     
        de_flatten(vector=weights_vector, model=model)
        # test_model(test_dataloader, model, loss_fn, DEVICE)
        Mytest(params_loaded, epoch=round_id, model=model, is_poison=False)
        Mytest_poison(params_loaded, epoch=round_id, model=model, is_poison=True)

    exit_flag = input("输入exit以结束：")
    while exit_flag != "exit":
        exit_flag = input()

    exit()
