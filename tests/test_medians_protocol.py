import threading
from random import random

from pefl_protocol.helpers import arr_enc, arr_dec
from test_basic import Configs, make_kgc_connector, make_cp_connector, make_sp, make_cp

from test_basic import *

# Configs.TRAINERS_COUNT = 10
Configs.TRAINERS_COUNT = 9
Configs.MODEL_LENGTH = 100

kgc_connector = make_kgc_connector()
cp_connector = make_cp_connector()
sp = make_sp(kgc_connector, cp_connector)
cp = make_cp(kgc_connector)

t = threading.Thread(target=cp.run)
t.start()

m = Configs.TRAINERS_COUNT
n = Configs.MODEL_LENGTH


print('Generating g.')
g = [[random() for _ in range(n)] for _ in range(m)]
for i in g:
    print(i)

print('Encrypting g.')
sp.gradient = [arr_enc(g[i], sp.pkc) for i in range(m)]

r = [random() for _ in range(sp.model_length)]
r = arr_enc(r, sp.pkc)

print('Running protocol.')
mg = sp.medians_protocol()

print('Decrypting the medians.')
mg = arr_dec(mg, cp.skc)

print('Checking the medians.')
tg = [[g[j][i] for j in range(m)] for i in range(n)]

for i in range(n):
    tg[i].sort()
    print(tg[i])

if m % 2 == 1:
    mg2 = [tg[i][m // 2] for i in range(n)]
else:
    mg2 = [(tg[i][m // 2 - 1] + tg[i][m // 2]) / 2 for i in range(n)]

for i in range(n):
    print(mg[i], mg2[i], abs(mg2[i] - mg[i]))
