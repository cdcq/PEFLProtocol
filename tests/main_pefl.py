import os
import sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(sys.path[0], "../pefl_protocol"))
# print(sys.path)

import random
import yaml
import torch
from time import sleep
from multiprocessing import Process
# from torch.multiprocessing import Process as gpuProcess
from torch.multiprocessing import Pool

from pefl_protocol.helpers import yield_accumulated_grads, flatten, de_flatten
from pefl_protocol.connector import Connector
from pefl_protocol.key_generator import KeyGenerator
from pefl_protocol.cloud_provider import CloudProvider
from pefl_protocol.service_provider import ServiceProvider
from pefl_protocol.trainer import Trainer

from ML_utils.get_data import get_train_dataset
from ML_utils.model import get_model
from ML_utils.local_update import local_update


DIR_OF_AUTH = "cert"
KGC_ADDR_PORT = ('127.0.0.1', 8700)
CP_ADDR_PORT = ('127.0.0.1', 8701)
SP_ADDR_PORT = ('127.0.0.1', 8702)
# KGC_ADDR_PORT = ('127.0.0.1', random.randint(1000, 9999))
# CP_ADDR_PORT = ('127.0.0.1', random.randint(1000, 9999))
# SP_ADDR_PORT = ('127.0.0.1', random.randint(1000, 9999))
TRAINERS_COUNT = 3
MAX_ROUND = 1000
DATASET_NAME = "mnist"
MODEL_NAME = "MLP"
MODEL_LENGTH = 633226
DEVICE = torch.device("cuda")


def register_users():
    dir_of_token = os.path.join(DIR_OF_AUTH, "token")
    if os.path.exists(dir_of_token):
        return
    os.mkdir(dir_of_token)
    registered_users_tokens = {
        "CP": {"Token": "CP", "Right": 0b10},
        "SP": {"Token": "SP", "Right": 0b00},
        "EDGE": {"Token": "EDGE", "Right": 0b01}
    }

    cp_token = {"User": "CP", "Token": "CP"}
    sp_token = {"User": "SP", "Token": "SP"}
    edge_token = {"User": "EDGE", "Token": "EDGE"}

    with open(os.path.join(DIR_OF_AUTH, "token", "registered_users.yml"), 'w') as f:
        yaml.safe_dump(registered_users_tokens, f)
    with open(os.path.join(DIR_OF_AUTH, "token", "cp.yml"), 'w') as f:
        yaml.safe_dump(cp_token, f)
    with open(os.path.join(DIR_OF_AUTH, "token", "sp.yml"), 'w') as f:
        yaml.safe_dump(sp_token, f)
    with open(os.path.join(DIR_OF_AUTH, "token", "edge.yml"), 'w') as f:
        yaml.safe_dump(edge_token, f)


def run_kgc():
    kgc = KeyGenerator(
        listening=KGC_ADDR_PORT,
        cert_path=os.path.join(DIR_OF_AUTH, "kgc.crt"),
        key_path=os.path.join(DIR_OF_AUTH, "kgc.key"),
        users_path=os.path.join(DIR_OF_AUTH, "token", "registered_users.yml")
    )
    print("KGC 正在启动")
    kgc.run()


def run_cloud_provider():
    key_generator = Connector(service=KGC_ADDR_PORT,
                              ca_path=os.path.join(DIR_OF_AUTH, "kgc.crt"),)
    cp = CloudProvider(
        listening=CP_ADDR_PORT,
        cert_path=os.path.join(DIR_OF_AUTH, 'cp.crt'),
        key_path=os.path.join(DIR_OF_AUTH, 'cp.key'),
        key_generator=key_generator,
        token_path=os.path.join(DIR_OF_AUTH, "token", "cp.yml")
    )
    print("CP 正在启动")
    cp.run()


def run_service_provider():
    key_generator = Connector(service=KGC_ADDR_PORT,
                              ca_path=os.path.join(DIR_OF_AUTH, "kgc.crt"))
    cloud_provider = Connector(service=CP_ADDR_PORT,
                               ca_path=os.path.join(DIR_OF_AUTH, "cp.crt"))
    sp = ServiceProvider(
        listening=SP_ADDR_PORT,
        cert_path=os.path.join(DIR_OF_AUTH, "sp.crt"),
        key_path=os.path.join(DIR_OF_AUTH, "sp.key"),
        key_generator=key_generator,
        cloud_provider=cloud_provider,
        token_path=os.path.join(DIR_OF_AUTH, "token", "sp.yml"),
        learning_rate=0.01,
        trainers_count=TRAINERS_COUNT,
        train_round=MAX_ROUND,
        model_length=MODEL_LENGTH
    )
    print("SP 正在启动")
    sp.run()


def run_edge():
    print("1111")
    key_generator = Connector(service=KGC_ADDR_PORT,
                              ca_path=os.path.join(DIR_OF_AUTH, "kgc.crt"))
    service_provider = Connector(service=SP_ADDR_PORT,
                                 ca_path=os.path.join(DIR_OF_AUTH, "sp.crt"))
    print("2222")
    edge = Trainer(
        key_generator=key_generator,
        service_provider=service_provider,
        token_path=os.path.join(DIR_OF_AUTH, "token", "edge.yml"),
        model_length=MODEL_LENGTH,
    )
    print("333")
    train_dataset = get_train_dataset(dataset="mnist", iid=True)
    print("444")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("5555")
    model = get_model(model_name="mlp", device=DEVICE)
    print("edge 正在启动")
    for _ in range(MAX_ROUND):
        grads_list, local_loss = local_update(model=model, dataloader=train_dataloader, device=DEVICE)
        grads_vector = flatten(yield_accumulated_grads(grads_list))
        weights_vector = edge.round_run(grads_vector)
        de_flatten(vector=weights_vector, model=model)


if __name__ == "__main__":
    # run_edge()
    # exit(0)

    # torch.multiprocessing.set_start_method('spawn')
    register_users()
    kgc_process = Process(target=run_kgc, args=(), daemon=True)
    kgc_process.start()
    sleep(3)
    cp_process = Process(target=run_cloud_provider, args=(), daemon=True)
    cp_process.start()
    sleep(2)
    sp_process = Process(target=run_service_provider, args=(), daemon=True)
    sp_process.start()
    sleep(2)

    exit_flag = input("输入exit以结束：")
    while exit_flag != "exit":
        exit_flag = input()

    exit()

    # 会报错
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    # # torch.multiprocessing.set_start_method('spawn', force=True)
    # p = Pool(TRAINERS_COUNT)
    # for i in range(TRAINERS_COUNT):
    #     p.apply_async(func=run_edge, args=())
    #
    # exit_flag = input("输入exit以结束：")
    # while exit_flag != "exit":
    #     exit_flag = input()
    # p.close()
    # p.join()
    # exit()
