import math
import numpy
import os
import threading
from random import random

from pefl_protocol.cloud_provider import CloudProvider
from pefl_protocol.connector import Connector
from pefl_protocol.helpers import arr_enc
from pefl_protocol.service_provider import ServiceProvider
from main_pefl import KGC_ADDR_PORT, CP_ADDR_PORT, SP_ADDR_PORT, DIR_OF_AUTH, \
    MAX_ROUND, TIME_OUT

TRAINERS_COUNT = 10
MODEL_LENGTH = 20

key_generator = Connector(
    service=KGC_ADDR_PORT,
    ca_path=os.path.join(DIR_OF_AUTH, "kgc.crt"),
    time_out=TIME_OUT
)
cloud_provider = Connector(
    service=CP_ADDR_PORT,
    ca_path=os.path.join(DIR_OF_AUTH, "kgc.crt"),
    time_out=TIME_OUT
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
    model_length=MODEL_LENGTH,
    time_out=TIME_OUT
)
cp = CloudProvider(
    listening=CP_ADDR_PORT,
    cert_path=os.path.join(DIR_OF_AUTH, 'cp.crt'),
    key_path=os.path.join(DIR_OF_AUTH, 'cp.key'),
    key_generator=key_generator,
    token_path=os.path.join(DIR_OF_AUTH, "token", "cp.yml"),
    time_out=TIME_OUT
)

t = threading.Thread(target=cp.run)
t.start()

gx = [random() for _ in range(MODEL_LENGTH)]
gy = [gx[i] + random() * 0.001 for i in range(MODEL_LENGTH)]
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
