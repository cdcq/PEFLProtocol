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
from ML_utils.test import test_model
from ML_utils.poison import exec_poisoning

TASK = 0
DATASET = {
    0: "mnist",
    1: "cifar-10",
    2: "CNNDetection",
}
MODEL = {
    0: "mlp",
    2:"resnet18_CNNDetect"
}
CALCULATE_MODEL_LENGTH = {
    0: 633226,
    2: 11177538    #修改过的resnet18
}
DATASET_NAME = DATASET[TASK]
MODEL_NAME = MODEL[TASK]
MODEL_LENGTH = CALCULATE_MODEL_LENGTH[TASK]

TRAINERS_COUNT = 5
MAX_ROUND = 1000
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='PPDL')
    # parser.add_argument('--params', default="ML_utils/utils/mnist_params.yaml", dest='params')
    # args = parser.parse_args()
    # with open(f'./{args.params}', 'r') as f:
    #     params_loaded = yaml.safe_load(f)

    model = get_model(model_name=MODEL_NAME, device=DEVICE)
    # if params_loaded['resumed_model']:
    #     if torch.cuda.is_available() :
    #         loaded_params = torch.load(f"saved_models/{params_loaded['resumed_model_name']}")
    #     else:
    #         loaded_params = torch.load(f"saved_models/{params_loaded['resumed_model_name']}", map_location='cpu')
    #     model.load_state_dict(loaded_params['state_dict'])
    #     start_epoch = loaded_params['epoch'] + 1
    #     params_loaded['lr'] = loaded_params.get('lr', params_loaded['lr'])
    #     print(f"Loaded parameters from saved model: LR is "
    #                 f"{params_loaded['lr']} and current epoch is {start_epoch}")
    # else:
    #     start_epoch = 0

    weights_vector = flatten(model.parameters())
    data_source = DatasetSource(dataset_name=DATASET_NAME)

    test_dataloader = data_source.get_test_dataloader()
    test_poison_dataloader = data_source.get_test_poison_loader()
    loss_fn = torch.nn.CrossEntropyLoss()

    edge_dataloaders = []
    for edge_id in range(TRAINERS_COUNT):
        edge_dataloader = data_source.get_train_dataloader(frac=0.3, iid=True)
        edge_dataloaders.append(edge_dataloader)


    for round_id in range(MAX_ROUND):
        grads_vector_sum = [.0] * MODEL_LENGTH

        for edge_id in range(TRAINERS_COUNT):
            de_flatten(vector=weights_vector, model=model)
            if exec_poisoning(round_id=round_id, edge_id=edge_id,
                              trainer_count=TRAINERS_COUNT, poison_freq=1, start_round=2):
                grads_list, local_loss = poison_local_update(model=model, dataloader=edge_dataloaders[edge_id],
                                                             trainer_count=TRAINERS_COUNT, edge_id=edge_id)
            else:
                grads_list, local_loss = local_update(model=model,
                                                      dataloader=edge_dataloaders[edge_id], loss_fn=loss_fn)
            print("Round = {:>3d}  edge_id = {:>2d} local_loss = {:.4f}".format(round_id, edge_id, local_loss))
            grads_vector = flatten(yield_accumulated_grads(grads_list))
            for dimension in range(MODEL_LENGTH):
                grads_vector_sum[dimension] += grads_vector[dimension]
        
        for dimension in range(MODEL_LENGTH):
            grads_vector_sum[dimension] /= TRAINERS_COUNT
            weights_vector[dimension] -= LEARNING_RATE * grads_vector_sum[dimension]
     
        de_flatten(vector=weights_vector, model=model)

        test_model(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=DEVICE,
                   epoch=round_id, is_poison=False)
        test_model(model=model, test_dataloader=test_poison_dataloader, loss_fn=loss_fn, device=DEVICE,
                   epoch=round_id, is_poison=True)

        if (round_id+1) % 5 == 0:
            model_save_path = os.path.join("saved_models", f"task_{TASK}", f"round_{round_id}.pt")
            torch.save(model.state_dict(), model_save_path)
