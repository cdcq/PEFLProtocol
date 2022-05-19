import os
from random import random

from pefl_protocol.cloud_provider import CloudProvider
from pefl_protocol.connector import Connector
from pefl_protocol.helpers import arr_enc, arr_dec
from pefl_protocol.service_provider import ServiceProvider
from main_pefl import KGC_ADDR_PORT, CP_ADDR_PORT, SP_ADDR_PORT, DIR_OF_AUTH, \
    MAX_ROUND

# TRAINERS_COUNT = 10
TRAINERS_COUNT = 9
MODEL_LENGTH = 5

key_generator = Connector(
    service=KGC_ADDR_PORT,
    ca_path=os.path.join(DIR_OF_AUTH, "kgc.crt")
)
cloud_provider = Connector(
    service=CP_ADDR_PORT,
    ca_path=os.path.join(DIR_OF_AUTH, "kgc.crt")
)
sp = ServiceProvider(
    listening=SP_ADDR_PORT,
    cert_path=os.path.join(DIR_OF_AUTH, "sp.crt"),
    key_path=os.path.join(DIR_OF_AUTH, "sp.key"),
    key_generator=key_generator,
    cloud_provider=cloud_provider,
    token_path=os.path.join(DIR_OF_AUTH, "token", "sp.yml"),
    learning_rate=0.01,
    trainers_count=TRAINERS_COUNT,
    train_round=MAX_ROUND,
    model_length=MODEL_LENGTH
)
cp = CloudProvider(
    listening=('127.0.0.1', 8703),
    cert_path=os.path.join(DIR_OF_AUTH, 'cp.crt'),
    key_path=os.path.join(DIR_OF_AUTH, 'cp.key'),
    key_generator=key_generator,
    token_path=os.path.join(DIR_OF_AUTH, "token", "cp.yml")
)

m = TRAINERS_COUNT
n = MODEL_LENGTH

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
