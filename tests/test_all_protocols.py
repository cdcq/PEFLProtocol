import math
import numpy
import threading
from phe import paillier
from random import random

from pefl_protocol.helpers import arr_enc, arr_dec
from test_basic import Configs, make_kgc_connector, make_sp_connector, make_cp_connector, \
    make_sp, make_cp, make_trainer

Configs.TRAINERS_COUNT = 1
Configs.MODEL_LENGTH = 10

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp_connector = make_sp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)

tr = make_trainer(kgc_connector, sp_connector)

model = [0.013801388442516327, -0.02055473066866398, 0.017501456663012505, 0.0080514345318079, -0.019214686006307602,
         -0.004485428333282471, 0.019524699077010155, 0.004331077914685011, 0.01430246327072382, 0.01712757907807827]

sp.model = model

t1 = threading.Thread(target=cp.run)
t1.start()

t2 = threading.Thread(target=sp.run)
t2.start()

gradient = [0.06068916991353035, 0.060690008103847504, 0.060690008103847504, 0.06068935617804527, 0.060690008103847504,
            0.06068935617804527, 0.060690008103847504, 0.06068900600075722, 0.06068916991353035, 0.060690008103847504]
model2 = tr.round_run(gradient)

print(arr_dec(sp.temp, cp.skc))
# print(arr_dec([paillier.EncryptedNumber(cp.skc.public_key, i) for i in sp.temp], cp.skc))
# print(sp.temp[0].exponent)
# print(paillier.EncryptedNumber(cp.skc.public_key, sp.temp[0]).exponent)

for i in range(len(model)):
    print(model[i], model2[i], abs(model[i] - model2[i]),
          abs(model[i] - model2[i] - gradient[i] * Configs.LEARNING_RATE))
