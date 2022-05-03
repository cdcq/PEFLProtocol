import math
import numpy
import ssl
from phe import paillier

from base_service import BaseService
from connector import Connector
from enums import Protocols, MessageItems
from helpers import send_obj, receive_obj, arr_enc, arr_dec
from key_generator import KeyRequester


class CloudProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector,
                 token_path: str,
                 time_out=10, max_connection=5,
                 precision=32):
        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection)
        KeyRequester.__init__(self, key_generator, token_path)

        self.precision = precision

        self.skc = self.request_key(Protocols.GET_SKC)
        self.pkx = self.request_key(Protocols.GET_PKX)
        self.trainers_count = 0
        self.mu = []
        self.dx = []

    def tcp_handler(self, conn: ssl.SSLSocket, address):
        conn.settimeout(self.time_out)
        print('Start a connection.')
        msg = receive_obj(conn)

        protocol = msg[MessageItems.PROTOCOL]
        if protocol == Protocols.SEC_MED:
            self.medians_handler(conn)
        elif protocol == Protocols.SEC_PER:
            self.pearson_handler(conn)
        elif protocol == Protocols.SEC_AGG:
            self.aggregate_handler(conn)

        msg = receive_obj(conn)
        if msg[MessageItems.DATA] == 'OK':
            conn.close()
            print('Closed a connection.')
        else:
            print('A protocol ended incorrectly. Protocol ID: {0}.'.format(protocol))

    def clout_init(self, conn: ssl.SSLSocket):
        msg = {
            MessageItems.PROTOCOL: Protocols.CLOUD_INIT
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        self.trainers_count = data['trainers_count']
        self.mu = [0 for _ in range(self.trainers_count)]

        msg = {
            MessageItems.PROTOCOL: Protocols.CLOUD_INIT
        }
        send_obj(conn, msg)

    def medians_handler(self, conn: ssl.SSLSocket):
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        r1 = msg[MessageItems.DATA]
        m, n = len(r1), len(r1[0])

        dc = [[paillier.EncryptedNumber(self.skc.public_key, r1[i][j])
               for j in range(n)] for i in range(m)]
        dx = [arr_dec(dc[i], self.skc, self.precision) for i in range(m)]

        # Acceleration.
        self.dx = dx

        n = len(dx[0])  # Attention.
        dt = [[dx[j][i] for j in range(m)] for i in range(n)]

        for i in range(n):
            dt[i].sort()

        if m % 2 == 1:
            dm = [dt[i][m // 2 + 1] for i in range(n)]
        else:
            dm = [(dt[i][m // 2] + dt[i][m // 2 + 1]) / 2 for i in range(n)]

        dc = arr_enc(dm, self.skc.public_key, self.precision)
        n = len(dc)
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED,
            MessageItems.DATA: [dc[i].ciphertext() for i in range(n)]
        }
        send_obj(conn, msg)

    def pearson_handler(self, conn: ssl.SSLSocket):
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        rx, ry, x_id = data['rx'], data['ry'], data['x_id']
        rx = [paillier.EncryptedNumber(self.skc.public_key, rx[i]) for i in range(len(rx))]
        ry = [paillier.EncryptedNumber(self.skc.public_key, ry[i]) for i in range(len(ry))]

        dx = arr_dec(rx, self.skc)
        dy = arr_dec(ry, self.skc)
        n = len(dx)

        rho = float(numpy.corrcoef(dx, dy)[0][1])

        small_number = 1e-6
        self.mu[x_id] = max(0.0, math.log((1 + rho) / (1 - rho + small_number)) - 0.5)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER
        }
        send_obj(conn, msg)

    def aggregate_handler(self, conn: ssl.SSLSocket):
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        nu = msg[MessageItems.DATA]

        m = self.trainers_count
        n = len(self.dx[0])
        sum_mu = sum(self.mu)
        k = [nu * self.mu[i] / (m * sum_mu) for i in range(m)]
        ex = [[k[i] * self.dx[i][j] for j in range(n)] for i in range(m)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG,
            MessageItems.DATA: {
                'ex': [arr_enc(ex[i], self.skc.public_key) for i in range(m)],
                'k': arr_enc(k, self.skc.public_key)
            }
        }
        send_obj(conn, msg)

    def exchange_handler(self, conn: ssl.SSLSocket):
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        omega = [paillier.EncryptedNumber(self.skc.public_key, data[i])
                 for i in range(len(data))]

        omega_x = arr_enc(arr_dec(omega, self.skc), self.pkx)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC,
            MessageItems.DATA: [omega_x[i].ciphertext() for i in range(len(omega_x))]
        }
        send_obj(conn, msg)
