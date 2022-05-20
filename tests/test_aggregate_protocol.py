import math
import numpy
import threading
from random import random

from pefl_protocol.helpers import arr_enc, arr_dec
from test_basic import Consts, make_kgc_connector, make_cp_connector, make_sp, make_cp

Consts.TRAINERS_COUNT = 5
Consts.MODEL_LENGTH = 5

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)

t = threading.Thread(target=cp.run)
t.start()

n = Consts.MODEL_LENGTH
m = Consts.TRAINERS_COUNT

g0 = [random() for _ in range(n)]

g = [g0.copy() for _ in range(m)]

for i in range(m):
    for j in range(n):
        g[i][j] = g[i][j] + 0.001 * random()

print('Encrypting g.')
sp.gradient = [arr_enc(g[i], sp.pkc) for i in range(m)]

model = [random() for _ in range(n)]
sp.model = arr_enc(model, sp.pkc)

print('Cloud init.')
sp.cloud_init()
print('Running medians protocol.')
mg = sp.medians_protocol()

print('Running pearson protocol.')
for i in range(sp.trainers_count):
    sp.pearson_protocol(sp.gradient[i], mg, i)

print('Running aggregate protocol')
sp.aggregate_protocol()

model2 = arr_dec(sp.model, cp.skc)

sum_mu = sum(cp.mu)
nu = sp.learning_rate
k = [nu * cp.mu[i] / (m * sum_mu) for i in range(m)]

model1 = model.copy()
for i in range(m):
    for j in range(n):
        model1[j] = model1[j] - g[i][j] * k[i]

for i in range(n):
    print(model[i], model1[i], model2[i], abs(model1[i] - model2[i]))
