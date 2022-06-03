from math import log
from phe import paillier
from random import random

from pefl_protocol import enc_utils
from test_basic import Configs

pub, prv = paillier.generate_paillier_keypair()

encryptor = enc_utils.Encryptor(pub, prv)

n = 10
m = 4

r = [random() * 1e3 for _ in range(n)] \
    + [random() for _ in range(n)] \
    + [random() * 1e-3 for _ in range(n)] \
    + [random() * -1 for _ in range(n)]

print('r:', r)

enc_r = encryptor.arr_enc(r)

dec_r = encryptor.arr_dec(enc_r, n * m)

print('arr_len:', len(enc_r))
for i in range(n * m):
    print(int(log(abs(r[i] - dec_r[i] + 1e-10), 10)), end=' ')

print('')

k = random()
k1 = int(k * 2 ** Configs.PRECISION)

mul_r = [enc_r[i] * k1 for i in range(len(enc_r))]
dec_mul = encryptor.arr_dec(mul_r, n * m)
dec_mul = [i / 2 ** Configs.PRECISION for i in dec_mul]
for i in range(n * m):
    print(int(log(abs(r[i] * k - dec_mul[i] + 1e-10), 10)), end=' ')

print('')

t = [random() for _ in range(n * m)]
t1 = encryptor.arr_enc(t)

pls_r = [enc_r[i] + t1[i] for i in range(len(enc_r))]
dec_pls = encryptor.arr_dec(pls_r, n * m)
for i in range(n * m):
    # print(r[i], t[i], dec_pls[i])
    print(int(log(abs(r[i] + t[i] - dec_pls[i] + 1e-10), 10)), end=' ')

print(' ')
