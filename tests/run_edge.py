import sys
import os
import time
import json
import torch

from torch.utils.tensorboard import SummaryWriter
from test_basic import make_sp_connector, make_kgc_connector, make_trainer
from configs import Configs

from pefl_protocol.helpers import flatten, yield_accumulated_grads, de_flatten
from ML_utils.get_data import DatasetSource
from ML_utils.model import get_model
from ML_utils.local_update import local_update, poison_local_update
from ML_utils.poison import exec_poisoning
from ML_utils.test import test_model

if __name__ == "__main__":
    edge_id = int(sys.argv[1])
    kgc_connector = make_kgc_connector()
    sp_connector = make_sp_connector()
    edge = make_trainer(kgc_connector, sp_connector, edge_id)

    model = get_model(model_name=Configs.MODEL_NAME, device=Configs.DEVICE)
    # 统一初始化
    with open(os.path.join("init_weights_vectors", f"{Configs.MODEL_NAME}.txt"), 'r') as read_file:
        init_weights_vector = json.load(read_file)
    de_flatten(vector=init_weights_vector, model=model)

    data_source = DatasetSource(dataset_name=Configs.DATASET_NAME, poison_label_swap=Configs.POISON_SWAP_LABEL)
    train_dataloader = data_source.get_train_dataloader(batch_size=Configs.BATCH_SIZE, frac=0.3)
    test_dataloader = data_source.get_test_dataloader()
    test_poison_dataloader = data_source.get_test_poison_loader()
    loss_fn = torch.nn.CrossEntropyLoss()

    if edge_id == Configs.SHOW_EDGE_ID:
        writer = SummaryWriter(log_dir=f"runs/task_{Configs.TASK}", flush_secs=60)
        writer.add_graph(model=model, input_to_model=next(iter(test_dataloader))[0].to(Configs.DEVICE))

    for round_id in range(Configs.MAX_ROUND):
        if exec_poisoning(round_id=round_id, edge_id=edge_id, trainer_count=Configs.TRAINERS_COUNT,
                          poison_freq=1, start_round=2):
            grads_list, local_loss = poison_local_update(model=model, previous_weights_vector=weights_vector,
                                                         dataloader=train_dataloader, alpha=Configs.ALPHA,
                                                         trainer_count=Configs.TRAINERS_COUNT, edge_id=edge_id,
                                                         round_id=round_id, task=Configs.TASK)
        else:
            grads_list, local_loss = local_update(model=model, dataloader=train_dataloader,
                                                  edge_id=edge_id, round_id=round_id)

        print(time.asctime(time.localtime(time.time())))
        grads_vector = flatten(yield_accumulated_grads(grads_list))
        weights_vector = edge.round_run(gradient=grads_vector)
        de_flatten(vector=weights_vector, model=model)
        print(time.asctime(time.localtime(time.time())))

        loss_normal, acc_normal = test_model(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn,
                                             device=Configs.DEVICE, epoch=round_id, is_poison=False, task=Configs.TASK)
        loss_poison, acc_poison = test_model(model=model, test_dataloader=test_poison_dataloader, loss_fn=loss_fn,
                                             device=Configs.DEVICE, epoch=round_id, is_poison=True, task=Configs.TASK)
        print("")

        if edge_id == Configs.SHOW_EDGE_ID:
            writer.add_scalars("Normal vs. Poison loss",
                               {"Normal": loss_normal, "Poison": loss_poison},
                               round_id)
            writer.add_scalars("Normal vs. Poison accuracy",
                               {"Normal": acc_normal, "Poison": acc_poison},
                               round_id)

            if (round_id+1) % 5 == 0:
                model_save_path = os.path.join("saved", "models", f"task_{Configs.TASK}", f"round_{round_id+1}.pt")
                torch.save(model.state_dict(), model_save_path)
