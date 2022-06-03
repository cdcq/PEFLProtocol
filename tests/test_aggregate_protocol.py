import math
import numpy
import threading
from random import random

from test_basic import Configs, make_kgc_connector, make_cp_connector, make_sp, make_cp

Configs.TRAINERS_COUNT = 5
Configs.MODEL_LENGTH = 5

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)

t = threading.Thread(target=cp.run)
t.start()

n = Configs.MODEL_LENGTH
m = Configs.TRAINERS_COUNT

g0 = [random() for _ in range(n)]

g = [g0.copy() for _ in range(m)]

for i in range(m):
    for j in range(n):
        g[i][j] = g[i][j] + 0.1 * random() - 0.05

print('Encrypting g.')
sp.gradient = [sp.enc_c.arr_enc(i) for i in g]

model = [random() for _ in range(n)]
sp.model = sp.enc_c.arr_enc(model)

print('Cloud init.')
sp.cloud_init()
print('Running medians protocol.')
mg = sp.medians_protocol()

print('Running pearson protocol.')
for i in range(sp.trainers_count):
    sp.pearson_protocol(sp.gradient[i], mg, i)

print('Running aggregate protocol')
sp.aggregate_protocol()

model2 = cp.enc_c.arr_dec(sp.model, n)

sum_mu = sum(cp.mu)
nu = sp.learning_rate
small_number = 1e-6
k = [nu * cp.mu[i] / (m * sum_mu + small_number) for i in range(m)]
print("k:", k)

model1 = model.copy()
for i in range(m):
    for j in range(n):
        model1[j] = model1[j] - g[i][j] * k[i]

for i in range(n):
    print(model[i], model1[i], model2[i], abs(model1[i] - model2[i]))
