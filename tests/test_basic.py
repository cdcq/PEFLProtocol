import os
import sys
import json

sys.path.append(os.path.join(sys.path[0], ".."))

from pefl_protocol.cloud_provider import CloudProvider
from pefl_protocol.connector import Connector
from pefl_protocol.key_generator import KeyGenerator
from pefl_protocol.service_provider import ServiceProvider
from pefl_protocol.trainer import Trainer
from config import Config


def make_kgc_connector():
    kgc_connector = Connector(
        service=Config.KGC_ADDR_PORT,
        ca_path=os.path.join(Config.DIR_OF_AUTH, "kgc.crt"),
        time_out=Config.TIME_OUT,
    )
    return kgc_connector


def make_sp_connector():
    sp_connector = Connector(
        service=Config.SP_ADDR_PORT,
        ca_path=os.path.join(Config.DIR_OF_AUTH, "kgc.crt"),
        time_out=Config.TIME_OUT
    )
    return sp_connector


def make_cp_connector():
    cp_connector = Connector(
        service=Config.CP_ADDR_PORT,
        ca_path=os.path.join(Config.DIR_OF_AUTH, "kgc.crt"),
        time_out=Config.TIME_OUT
    )
    return cp_connector


def make_kgc():
    kgc = KeyGenerator(
        listening=Config.KGC_ADDR_PORT,
        cert_path=os.path.join(Config.DIR_OF_AUTH, "kgc.crt"),
        key_path=os.path.join(Config.DIR_OF_AUTH, "kgc.key"),
        users_path=os.path.join(Config.DIR_OF_AUTH, "token", "registered_users.yml")
    )
    return kgc


def make_sp(kgc_connector: Connector, cp_connector: Connector) -> ServiceProvider:
    with open(os.path.join("init_weights_vectors", f"task_{Config.TASK}.txt"), 'r') as read_file:
        init_weights_vector = json.load(read_file)

    sp = ServiceProvider(
        listening=Config.SP_ADDR_PORT,
        cert_path=os.path.join(Config.DIR_OF_AUTH, "sp.crt"),
        key_path=os.path.join(Config.DIR_OF_AUTH, "sp.key"),
        key_generator=kgc_connector,
        cloud_provider=cp_connector,
        token_path=os.path.join(Config.DIR_OF_AUTH, "token", "sp.yml"),
        model = init_weights_vector,
        learning_rate=0.01,
        trainers_count=Config.TRAINERS_COUNT,
        train_round=Config.MAX_ROUND,
        model_length=Config.MODEL_LENGTH,
        time_out=Config.TIME_OUT,
        precision=Config.PRECISION
    )
    return sp


def make_cp(kgc_connector: Connector) -> CloudProvider:
    cp = CloudProvider(
        listening=Config.CP_ADDR_PORT,
        cert_path=os.path.join(Config.DIR_OF_AUTH, 'cp.crt'),
        key_path=os.path.join(Config.DIR_OF_AUTH, 'cp.key'),
        key_generator=kgc_connector,
        token_path=os.path.join(Config.DIR_OF_AUTH, "token", "cp.yml"),
        time_out=Config.TIME_OUT,
        precision=Config.PRECISION
    )
    return cp


def make_trainer(kgc_connector: Connector, sp_connector: Connector):
    trainer = Trainer(
        key_generator=kgc_connector,
        service_provider=sp_connector,
        token_path=os.path.join(Config.DIR_OF_AUTH, "token", "edge.yml"),
        model_length=Config.MODEL_LENGTH,
        precision=Config.PRECISION
    )
    return trainer
