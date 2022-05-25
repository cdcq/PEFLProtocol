import os
import sys
import torch
sys.path.append(os.path.join(sys.path[0], ".."))
from torch.utils.tensorboard import SummaryWriter

from pefl_protocol.helpers import flatten, yield_accumulated_grads, de_flatten
from ML_utils.get_data import DatasetSource
from ML_utils.model import get_model
from ML_utils.local_update import local_update, poison_local_update
from ML_utils.test import test_model
from ML_utils.poison import exec_poisoning
from configs import Configs


if __name__ == "__main__":

    model = get_model(model_name=Configs.MODEL_NAME, device=Configs.DEVICE)
    weights_vector = flatten(model.parameters())
    data_source = DatasetSource(dataset_name=Configs.DATASET_NAME)

    test_dataloader = data_source.get_test_dataloader()
    test_poison_dataloader = data_source.get_test_poison_loader()
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(f'runs/task_{Configs.TASK}', flush_secs=60)

    edge_dataloaders = []
    for edge_id in range(Configs.TRAINERS_COUNT):
        edge_dataloader = data_source.get_train_dataloader(frac=0.3, iid=True)
        edge_dataloaders.append(edge_dataloader)


    for round_id in range(Configs.MAX_ROUND):
        grads_vector_sum = [.0] * Configs.MODEL_LENGTH

        for edge_id in range(Configs.TRAINERS_COUNT):
            de_flatten(vector=weights_vector, model=model)
            if exec_poisoning(round_id=round_id, edge_id=edge_id,
                              trainer_count=Configs.TRAINERS_COUNT, poison_freq=1, start_round=2):
                grads_list, local_loss = poison_local_update(model=model, dataloader=edge_dataloaders[edge_id],
                                                             trainer_count=Configs.TRAINERS_COUNT, edge_id=edge_id,
                                                             round_id=round_id, task=Configs.TASK)
            else:
                grads_list, local_loss = local_update(model=model, dataloader=edge_dataloaders[edge_id],
                                                      edge_id=edge_id, round_id=round_id)

            grads_vector = flatten(yield_accumulated_grads(grads_list))
            for dimension in range(Configs.MODEL_LENGTH):
                grads_vector_sum[dimension] += grads_vector[dimension]
        
        for dimension in range(Configs.MODEL_LENGTH):
            grads_vector_sum[dimension] /= Configs.TRAINERS_COUNT
            weights_vector[dimension] -= Configs.LEARNING_RATE * grads_vector_sum[dimension]
     
        de_flatten(vector=weights_vector, model=model)

        test_model(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=Configs.DEVICE,
                   epoch=round_id, is_poison=False, task=Configs.TASK)
        test_model(model=model, test_dataloader=test_poison_dataloader, loss_fn=loss_fn, device=Configs.DEVICE,
                   epoch=round_id, is_poison=True, task=Configs.TASK)
        print("")

        if (round_id+1) % 5 == 0:
            model_save_path = os.path.join("saved", "models", f"task_{Configs.TASK}", f"round_{round_id + 1}.pt")
            torch.save(model.state_dict(), model_save_path)
