import threading
from random import random

from pefl_protocol.helpers import arr_enc, arr_dec

from test_basic import Configs, make_kgc_connector, make_cp_connector, make_sp_connector, \
    make_sp, make_cp, make_trainer

Configs.TRAINERS_COUNT = 10
Configs.MODEL_LENGTH = 20

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp_connector = make_sp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)
trainer = make_trainer(kgc_connector, sp_connector)

t = threading.Thread(target=cp.run)
t.start()

n = Configs.MODEL_LENGTH
m = Configs.TRAINERS_COUNT

model = [random() for _ in range(n)]
print('Encrypting the model.')
sp.model = arr_enc(model, sp.pkc)

print('Start exchange protocol.')
sp.exchange_protocol()

print('Decrypting the model.')
model2 = arr_dec(sp.model_x, trainer.skx)

for i in range(n):
    print(model[i], model2[i], abs(model[i] - model2[i]))
