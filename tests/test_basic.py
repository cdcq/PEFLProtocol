import os
import sys

sys.path.append(os.path.join(sys.path[0], ".."))

import json
from random import random

from pefl_protocol.cloud_provider import CloudProvider
from pefl_protocol.connector import Connector
from pefl_protocol.key_generator import KeyGenerator
from pefl_protocol.service_provider import ServiceProvider
from pefl_protocol.trainer import Trainer
from configs import Configs


def make_kgc_connector():
    kgc_connector = Connector(
        service=Configs.KGC_ADDR_PORT,
        ca_path=os.path.join(Configs.DIR_OF_AUTH, "kgc.crt"),
        time_out=Configs.TIME_OUT,
    )
    return kgc_connector


def make_sp_connector():
    sp_connector = Connector(
        service=Configs.SP_ADDR_PORT,
        ca_path=os.path.join(Configs.DIR_OF_AUTH, "kgc.crt"),
        time_out=Configs.TIME_OUT,
    )
    return sp_connector


def make_cp_connector():
    cp_connector = Connector(
        service=Configs.CP_ADDR_PORT,
        ca_path=os.path.join(Configs.DIR_OF_AUTH, "kgc.crt"),
        time_out=Configs.TIME_OUT,
    )
    return cp_connector


def make_kgc():
    kgc = KeyGenerator(
        listening=Configs.KGC_ADDR_PORT,
        cert_path=os.path.join(Configs.DIR_OF_AUTH, "kgc.crt"),
        key_path=os.path.join(Configs.DIR_OF_AUTH, "kgc.key"),
        users_path=os.path.join(Configs.DIR_OF_AUTH, "token", "registered_users.yml"),
        key_size=Configs.KEY_SIZE,
    )
    return kgc


def make_sp(kgc_connector: Connector, cp_connector: Connector) -> ServiceProvider:
    if os.path.exists(os.path.join("init_weights_vectors", f"task_{Configs.TASK}.txt")):
        with open(os.path.join("init_weights_vectors", f"task_{Configs.TASK}.txt"), 'r') as read_file:
            init_weights_vector = json.load(read_file)
    else:
        init_weights_vector = [random() for _ in range(Configs.MODEL_LENGTH)]

    sp = ServiceProvider(
        listening=Configs.SP_ADDR_PORT,
        cert_path=os.path.join(Configs.DIR_OF_AUTH, "sp.crt"),
        key_path=os.path.join(Configs.DIR_OF_AUTH, "sp.key"),
        key_generator=kgc_connector,
        cloud_provider=cp_connector,
        token_path=os.path.join(Configs.DIR_OF_AUTH, "token", "sp.yml"),
        init_model=init_weights_vector,
        learning_rate=0.01,
        trainers_count=Configs.TRAINERS_COUNT,
        train_round=Configs.MAX_ROUND,
        model_length=Configs.MODEL_LENGTH,
        time_out=Configs.TIME_OUT,
        precision=Configs.PRECISION,
        mu_export_path=os.path.join("saved", "mus", f"task_{Configs.TASK}", "mu_table.csv")
    )
    return sp


def make_cp(kgc_connector: Connector) -> CloudProvider:
    cp = CloudProvider(
        listening=Configs.CP_ADDR_PORT,
        cert_path=os.path.join(Configs.DIR_OF_AUTH, 'cp.crt'),
        key_path=os.path.join(Configs.DIR_OF_AUTH, 'cp.key'),
        key_generator=kgc_connector,
        token_path=os.path.join(Configs.DIR_OF_AUTH, "token", "cp.yml"),
        time_out=Configs.TIME_OUT,
        precision=Configs.PRECISION,
    )
    return cp


def make_trainer(kgc_connector: Connector, sp_connector: Connector, edge_id: int):
    trainer = Trainer(
        key_generator=kgc_connector,
        service_provider=sp_connector,
        token_path=os.path.join(Configs.DIR_OF_AUTH, "token", f"edge{edge_id}.yml"),
        model_length=Configs.MODEL_LENGTH,
        precision=Configs.PRECISION,
    )
    return trainer
