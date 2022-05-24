import sys
import time
import os
import json

from test_basic import make_sp_connector, make_kgc_connector, make_trainer
from configs import Configs

from pefl_protocol.helpers import flatten, yield_accumulated_grads, de_flatten
from ML_utils.get_data import DatasetSource
from ML_utils.model import get_model
from ML_utils.local_update import local_update, poison_local_update
from ML_utils.poison import exec_poisoning

if __name__ == "__main__":
    kgc_connector = make_kgc_connector()
    sp_connector = make_sp_connector()
    edge = make_trainer(kgc_connector, sp_connector)

    edge_id = int(sys.argv[1])
    model = get_model(model_name=Configs.MODEL_NAME, device=Configs.DEVICE)
    # 统一初始化
    with open(os.path.join("init_weights_vectors", f"task_{Configs.TASK}.txt"), 'r') as read_file:
        init_weights_vector = json.load(read_file)
    de_flatten(vector=init_weights_vector, model=model)

    data_source = DatasetSource(dataset_name=Configs.DATASET_NAME, poison_label_swap=Configs.POISON_SWAP_LABEL)
    train_dataloader = data_source.get_train_dataloader(batch_size=Configs.BATCH_SIZE, frac=0.3)

    for round_id in range(Configs.MAX_ROUND):
        if exec_poisoning(round_id=round_id, edge_id=edge_id, trainer_count=Configs.TRAINERS_COUNT,
                          poison_freq=1, start_round=2):
            grads_list, local_loss = poison_local_update(model=model, dataloader=train_dataloader,
                                                         trainer_count=Configs, edge_id=edge_id,
                                                         round_id=round_id, task=Configs.TASK)
        else:
            grads_list, local_loss = local_update(model=model, dataloader=train_dataloader,
                                                  edge_id=edge_id, round_id=round_id)

        print(time.asctime(time.localtime(time.time())))
        grads_vector = flatten(yield_accumulated_grads(grads_list))
        weights_vector = edge.round_run(gradient=grads_vector)
        de_flatten(vector=weights_vector, model=model)
        print(time.asctime(time.localtime(time.time())))
