import os
import threading
from random import random

from pefl_protocol.trainer import Trainer
from test_basic import Consts, make_kgc_connector, make_cp_connector, make_sp_connector, make_sp

Consts.TRAINERS_COUNT = 3
Consts.MODEL_LENGTH = 10

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp_connector = make_sp_connector()
sp = make_sp(kgc_connector, cp_connector)

m = Consts.TRAINERS_COUNT
n = Consts.MODEL_LENGTH

tokens = ['edge0.yml', 'edge1.yml', 'edge2.yml']
trainers = []
for i in range(m):
    trainer = Trainer(
        key_generator=kgc_connector,
        service_provider=sp_connector,
        token_path=os.path.join(Consts.DIR_OF_AUTH, 'token', tokens[i]),
        model_length=Consts.MODEL_LENGTH,
    )
    trainers.append(trainer)

t0 = threading.Thread(target=sp.round_ready)
t0.start()

g = [[random() for _ in range(n)] for _ in range(m)]
t = []
for i in range(m):
    trainers[i].gradient = g[i]
    t.append(threading.Thread(target=trainers[i].round_ready))
    # t[i].start()
