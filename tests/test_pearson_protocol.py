import math
import numpy
import threading
from random import random

from pefl_protocol.helpers import arr_enc
from test_basic import Consts, make_kgc_connector, make_cp_connector, make_sp, make_cp

Consts.TRAINERS_COUNT = 10
Consts.MODEL_LENGTH = 20

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)

t = threading.Thread(target=cp.run)
t.start()

n = Consts.MODEL_LENGTH
m = Consts.TRAINERS_COUNT

gx = [random() for _ in range(n)]
gy = [gx[i] + random() * 0.001 for i in range(n)]
print('Encrypting.')
dx = arr_enc(gx, sp.pkc)
dy = arr_enc(gy, sp.pkc)

print('Cloud init.')
sp.cloud_init()

print('Start protocol.')
sp.pearson_protocol(dx, dy, 0)

print('calculate rho.')
rho = float(numpy.corrcoef(gx, gy)[0][1])

small_number = 1e-6
mu = max(0.0, math.log((1 + rho) / (1 - rho + small_number)) - 0.5)

print(mu, cp.mu[0], abs(mu - cp.mu[0]))
