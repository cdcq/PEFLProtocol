import os
import threading
from random import random
from time import sleep

from pefl_protocol.enc_utils import arr_enc
from pefl_protocol.trainer import Trainer
from test_basic import Configs, make_kgc_connector, make_cp_connector, make_sp_connector, make_sp, make_cp

Configs.TRAINERS_COUNT = 3
Configs.MODEL_LENGTH = 10

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp_connector = make_sp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)

m = Configs.TRAINERS_COUNT
n = Configs.MODEL_LENGTH

tokens = ['edge0.yml', 'edge1.yml', 'edge2.yml']
trainers = []
for i in range(m):
    trainer = Trainer(
        key_generator=kgc_connector,
        service_provider=sp_connector,
        token_path=os.path.join(Configs.DIR_OF_AUTH, 'token', tokens[i]),
        model_length=Configs.MODEL_LENGTH,
    )
    trainers.append(trainer)

model = [random() for _ in range(n)]
sp.model_x = arr_enc(model, sp.pkx)

t0 = threading.Thread(target=sp.distribute_model)
t0.start()

for i in range(m):
    model2 = trainers[i].get_model()
    for j in range(n):
        print(abs(model[j] - model2[j]), end=' ')
    print('')
