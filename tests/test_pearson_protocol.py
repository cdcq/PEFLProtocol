import math
import numpy
import threading
from random import random

from test_basic import Configs, make_kgc_connector, make_cp_connector, make_sp, make_cp

Configs.TRAINERS_COUNT = 10
Configs.MODEL_LENGTH = 20

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)

t = threading.Thread(target=cp.run)
t.start()

n = Configs.MODEL_LENGTH
m = Configs.TRAINERS_COUNT

gx = [random() for _ in range(n)]
gy = [gx[i] + random() * 0.001 for i in range(n)]
print('Encrypting.')
dx = sp.enc_c.arr_enc(gx)
dy = sp.enc_c.arr_enc(gy)

print('Cloud init.')
sp.cloud_init()

print('Start protocol.')
sp.pearson_protocol(dx, dy, 0)

print('calculate rho.')
rho = float(numpy.corrcoef(gx, gy)[0][1])

small_number = 1e-10
reject_number = 0.6
mu = max(0.0, math.log((1 + rho) / (1 - rho + small_number)) - reject_number)

print(mu, cp.mu[0], abs(mu - cp.mu[0]))
