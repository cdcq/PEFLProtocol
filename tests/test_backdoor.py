import os
import sys
import torch
sys.path.append(os.path.join(sys.path[0], ".."))

from pefl_protocol.helpers import flatten, yield_accumulated_grads, de_flatten
from ML_utils.get_data import get_train_dataset, get_test_dataset
from ML_utils.model import get_model
from ML_utils.local_update import local_update, posioned_local_update


TRAINERS_COUNT = 3
MAX_ROUND = 1000
DATASET_NAME = "mnist"
MODEL_NAME = "mlp"
MODEL_LENGTH = 633226
LEARNING_RATE = 0.1
DEVICE = torch.device("cuda")

def test_model(test_set, net, loss_fn, device):
    net.eval()
    total_loss = 0
    correct = 0
    for x, y in test_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = net(x)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #item()可将tensor数转化为一般数
    t_loss = total_loss / len(test_set)
    t_accuracy = correct / len(test_set.dataset)
    print("test_loss = {:>4f} test_accuracy = {:.2%}".format(t_loss, t_accuracy))

if __name__ == "__main__":

    model = get_model(model_name=MODEL_NAME, device=DEVICE)
    print(model)
    weights_vector = [.0] * MODEL_LENGTH
    test_dataset = get_test_dataset(dataset=DATASET_NAME)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    adversary = [False, False, False]
    assert len(adversary) == TRAINERS_COUNT
    edge_dataloaders = []
    for edge_id in range(TRAINERS_COUNT):
        if adversary[edge_id] == False:
            edge_dataset = get_train_dataset(dataset="mnist", iid=True)
        else:
            edge_dataset = get_train_dataset(dataset="posioned_mnist", iid=True)
        edge_dataloader = torch.utils.data.DataLoader(edge_dataset, batch_size=32, shuffle=True)
        edge_dataloaders.append(edge_dataloader)

    for round_id in range(MAX_ROUND):
        grads_vector_sum = [.0] * MODEL_LENGTH
        de_flatten(vector=weights_vector, model=model)
        for edge_id in range(TRAINERS_COUNT):
            if adversary[edge_id] == False:
                grads_list, local_loss = local_update(model=model, dataloader=edge_dataloaders[edge_id])
            else:
                grads_list, local_loss = posioned_local_update(model=model, dataloader=edge_dataloaders[edge_id])
            print("Round = {:>3d}  edge_id = {:>2d} local_loss = {:.4f}".format(round_id, edge_id, local_loss))
            grads_vector = flatten(yield_accumulated_grads(grads_list))
            for dimension in range(MODEL_LENGTH):
                grads_vector_sum[dimension] += grads_vector[dimension]

        for dimension in range(MODEL_LENGTH):
            grads_vector_sum[dimension] /= TRAINERS_COUNT
            weights_vector[dimension] -= LEARNING_RATE * grads_vector_sum[dimension]

        de_flatten(vector=weights_vector, model=model)

        test_model(test_dataloader, model, loss_fn, DEVICE)


    exit_flag = input("输入exit以结束：")
    while exit_flag != "exit":
        exit_flag = input()

    exit()


