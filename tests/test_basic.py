import os
import sys
from random import randint

sys.path.append(os.path.join(sys.path[0], ".."))

from pefl_protocol.cloud_provider import CloudProvider
from pefl_protocol.connector import Connector
from pefl_protocol.key_generator import KeyGenerator
from pefl_protocol.service_provider import ServiceProvider
from pefl_protocol.trainer import Trainer


class Consts:
    DIR_OF_AUTH = "cert"
    KGC_ADDR_PORT = ('127.0.0.1', 8700)
    # CP_ADDR_PORT = ('127.0.0.1', 8701)
    # SP_ADDR_PORT = ('127.0.0.1', 8702)
    # KGC_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    CP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    SP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    TRAINERS_COUNT = 3
    MAX_ROUND = 1000
    DATASET_NAME = "mnist"
    MODEL_NAME = "mlp"
    MODEL_LENGTH = 633226
    LEARNING_RATE = 0.1
    TIME_OUT = 300


def make_kgc_connector():
    kgc_connector = Connector(
        service=Consts.KGC_ADDR_PORT,
        ca_path=os.path.join(Consts.DIR_OF_AUTH, "kgc.crt"),
        time_out=Consts.TIME_OUT
    )
    return kgc_connector


def make_sp_connector():
    sp_connector = Connector(
        service=Consts.SP_ADDR_PORT,
        ca_path=os.path.join(Consts.DIR_OF_AUTH, "kgc.crt"),
        time_out=Consts.TIME_OUT
    )
    return sp_connector


def make_cp_connector():
    cp_connector = Connector(
        service=Consts.CP_ADDR_PORT,
        ca_path=os.path.join(Consts.DIR_OF_AUTH, "kgc.crt"),
        time_out=Consts.TIME_OUT
    )
    return cp_connector


def make_kgc():
    kgc = KeyGenerator(
        listening=Consts.KGC_ADDR_PORT,
        cert_path=os.path.join(Consts.DIR_OF_AUTH, "kgc.crt"),
        key_path=os.path.join(Consts.DIR_OF_AUTH, "kgc.key"),
        users_path=os.path.join(Consts.DIR_OF_AUTH, "token", "registered_users.yml")
    )
    return kgc


def make_sp(kgc_connector: Connector, cp_connector: Connector) -> ServiceProvider:
    sp = ServiceProvider(
        listening=Consts.SP_ADDR_PORT,
        cert_path=os.path.join(Consts.DIR_OF_AUTH, "sp.crt"),
        key_path=os.path.join(Consts.DIR_OF_AUTH, "sp.key"),
        key_generator=kgc_connector,
        cloud_provider=cp_connector,
        token_path=os.path.join(Consts.DIR_OF_AUTH, "token", "sp.yml"),
        learning_rate=0.01,
        trainers_count=Consts.TRAINERS_COUNT,
        train_round=Consts.MAX_ROUND,
        model_length=Consts.MODEL_LENGTH,
        time_out=Consts.TIME_OUT
    )
    return sp


def make_cp(kgc_connector: Connector) -> CloudProvider:
    cp = CloudProvider(
        listening=Consts.CP_ADDR_PORT,
        cert_path=os.path.join(Consts.DIR_OF_AUTH, 'cp.crt'),
        key_path=os.path.join(Consts.DIR_OF_AUTH, 'cp.key'),
        key_generator=kgc_connector,
        token_path=os.path.join(Consts.DIR_OF_AUTH, "token", "cp.yml"),
        time_out=Consts.TIME_OUT
    )
    return cp


def make_trainer(kgc_connector: Connector, sp_connector: Connector):
    trainer = Trainer(
        key_generator=kgc_connector,
        service_provider=sp_connector,
        token_path=os.path.join(Consts.DIR_OF_AUTH, "token", "edge.yml"),
        model_length=Consts.MODEL_LENGTH,
    )
    return trainer
