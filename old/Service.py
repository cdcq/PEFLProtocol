import ssl
import json
from random import getrandbits
from phe import paillier

from Connector import Connector
from PHEHelper import encrypted2dict, dict2encrypted


class ServiceProvider(Connector):
    def __init__(self, cloud: (str, int),
                 time_out=10, gradient_size=1000,
                 public_key: paillier.PaillierPublicKey = None,
                 context: ssl.SSLContext = None):
        Connector.__init__(cloud, time_out, public_key, context)

        self.gradient_size = gradient_size

    def medians_protocol(self, g: [[paillier.EncryptedNumber]], exponent=-32) \
            -> [paillier.EncryptedNumber]:
        m, n = g.shape
        r = [self.public_key.encrypt(getrandbits(1024)).decrease_exponent_to(exponent)
             for _ in range(n)]
        r1 = [[encrypted2dict(g[i][j] + r[j]) for i in range(m)] for j in range(n)]

        sock = self.start_connect()

        msg = {
            'Protocol': 'SecMed',
            'Data': r1
        }
        sock.send(json.dumps(msg).encode())

        msg = sock.recv(self.gradient_size * self.LENGTH_OF_ENCRYPTED + 100)
        msg = json.loads(msg)
        dc = [dict2encrypted(msg['Data'][i], self.public_key) for i in range(n)]
        gm = [dc[i] - r[i] for i in range(n)]

        sock.close()
        return gm

    def pearson_protocol(self, gx: [paillier.EncryptedNumber], gy: [paillier.EncryptedNumber],
                         exponent=-32) -> float:
        n = len(gx)
        r1 = [self.public_key.encrypt(getrandbits(1024)).decrease_exponent_to(exponent)
              for _ in range(n)]
        r2 = [self.public_key.encrypt(getrandbits(1024)).decrease_exponent_to(exponent)
              for _ in range(n)]
        rx = [encrypted2dict(gx[i] * r1[i]) for i in range(n)]
        ry = [encrypted2dict(gy[i] * r2[i]) for i in range(n)]

        sock = self.start_connect()

        msg = {
            'Protocol': 'SecPear',
            'Data': {
                'Rx': rx,
                'Ry': ry
            }
        }
        sock.send(json.dumps(msg).encode())

        msg = sock.recv(100)
        msg = json.loads(msg)
        return msg['Data']
