"""This is the cloud provider part of the PEFL protocol

The class is inherit base service class and key requester class, so it is able to listen TCP
connection request and request key from the Key Generation Center.
There are two points to notice. First is the key generator is a "Connector" object and you
need to generate the object by your self.
Second is that the run function is define at the base service class so you couldn't find it
at this class. but you can just run the "run" function and the TCP listener will at work.

Typical usage example:

cp = CloudProvider(listening, cert_path, key_path, key_generator, token_path)
cp.run()

"""

import math
import numpy
import ssl
from phe import paillier

from pefl_protocol.base_service import BaseService
from pefl_protocol.connector import Connector
from pefl_protocol.consts import Protocols, MessageItems
from pefl_protocol.helpers import send_obj, receive_obj, arr_enc, arr_dec
from pefl_protocol.key_generator import KeyRequester


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

    def tcp_handler(self, conn: ssl.SSLSocket, address):
        """

        This function rewrite the function in super class, analysis the protocol of the
        TCP request and chose the correct processing function.

        :param conn: the TCP connection to be handled.
        :param address: no use.
        :return: None.
        """

        conn.settimeout(self.time_out)
        print('Start a connection.')
        msg = receive_obj(conn)

        protocol = msg[MessageItems.PROTOCOL]
        if protocol == Protocols.CLOUD_INIT:
            self.clout_init(conn)
        elif protocol == Protocols.SEC_MED:
            self.medians_handler(conn)
        elif protocol == Protocols.SEC_PER:
            self.pearson_handler(conn)
        elif protocol == Protocols.SEC_AGG:
            self.aggregate_handler(conn)
        elif protocol == Protocols.SEC_EXC:
            self.exchange_handler(conn)

        msg = receive_obj(conn)
        if msg[MessageItems.DATA] == 'OK':
            conn.close()
            print('Closed a connection.')
        else:
            print('A protocol ended incorrectly. Protocol ID: {0}.'.format(protocol))

    def clout_init(self, conn: ssl.SSLSocket):
        """

        This function make some initialization of the cloud provider before a train
        round start.
        The initialization is contained:
            - Get the count of trainers.

        :param conn: the TCP connection that be used to initialize.
        :return: None.
        """

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
        """

        This function realize the "SecMed" procedure in the PEFL protocol.
        The CP will get encrypted gradient vectors of every trainers from SP
        and send a encrypted medians vector of these vectors to SP.

        :param conn: the TCP connection that be used to make this procedure.
        :return: None.
        """

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        r1 = msg[MessageItems.DATA]
        # TODO: Get m and n by the length of the array is not a good idea.
        m, n = len(r1), len(r1[0])

        dc = [[paillier.EncryptedNumber(self.skc.public_key, r1[i][j])
               for j in range(n)] for i in range(m)]
        dx = [arr_dec(dc[i], self.skc, self.precision) for i in range(m)]

        n = len(dx[0])  # Attention.
        # Transpose the dx matrix. It is convenient for sorting the column vectors in
        # matrix dx and then calculate the medians in a column vector.
        dt = [[dx[j][i] for j in range(m)] for i in range(n)]

        for i in range(n):
            dt[i].sort()

        if m % 2 == 1:
            dm = [dt[i][m // 2] for i in range(n)]
        else:
            dm = [(dt[i][m // 2 - 1] + dt[i][m // 2]) / 2 for i in range(n)]

        dc = arr_enc(dm, self.skc.public_key, self.precision)
        n = len(dc)
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED,
            MessageItems.DATA: [dc[i].ciphertext() for i in range(n)]
        }
        send_obj(conn, msg)

    def pearson_handler(self, conn: ssl.SSLSocket):
        """

        This function realize the "SecPear" procedure in PEFL protocol.
        The CP will get any one encrypted vector of the trainers and the encrypted median
        vector from SP. Then It will calculate the correlation coefficient of them and save
        the coefficient by itself.

        :param conn: the TCP connection that be used to make this procedure.
        :return: None.
        """

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        rx, ry, x_id = data['rx'], data['ry'], data['x_id']
        rx = [paillier.EncryptedNumber(self.skc.public_key, rx[i]) for i in range(len(rx))]
        ry = [paillier.EncryptedNumber(self.skc.public_key, ry[i]) for i in range(len(ry))]

        dx = arr_dec(rx, self.skc, self.precision)
        dy = arr_dec(ry, self.skc, self.precision)
        n = len(dx)

        rho = float(numpy.corrcoef(dx, dy)[0][1])

        small_number = 1e-6
        # This is the weight formula mentioned in PEFL paper.
        self.mu[x_id] = max(0.0, math.log((1 + rho) / (1 - rho + small_number)) - 0.5)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER
        }
        send_obj(conn, msg)

    def aggregate_handler(self, conn: ssl.SSLSocket):
        """

        This function realize the "SecAgg" procedure in PEFL protocol.
        The CP will get the learning rate nu from SP and calculate the next round model using
        the gradient saved at "SecMed" procedure. But the model is added noise so the CP send
        the coefficient of the model vector to SP and SP will minus the noise by homomorphic
        operation.

        :param conn: the TCP connection that be used to make this procedure.
        :return: None.
        """

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        nu = msg[MessageItems.DATA]['nu']
        g = msg[MessageItems.DATA]['g']

        g = [[paillier.EncryptedNumber(self.skc.public_key, i) for i in j] for j in g]
        gm = [arr_dec(i, self.skc, self.precision) for i in g]
        m = self.trainers_count
        n = len(gm[0])

        sum_mu = sum(self.mu)
        # "k" is the aggregate formula mentioned in PEFL paper.

        small_number = 1e-6
        k = [nu * self.mu[i] / (m * sum_mu + small_number) for i in range(m)]
        ex = [[k[i] * gm[i][j] for j in range(n)] for i in range(m)]

        # TODO: Here is an ugly fix for a bug, need to change!
        # kc = arr_enc(k, self.skc.public_key, self.precision)
        exc = [arr_enc(ex[i], self.skc.public_key, self.precision) for i in range(m)]
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG,
            MessageItems.DATA: {
                'ex': [[i.ciphertext() for i in j] for j in exc],
                # 'k': [i.ciphertext() for i in kc]
                # Attention, k in here is plain text!
                'k': k
            }
        }
        print("mu =", self.mu)
        send_obj(conn, msg)

    def exchange_handler(self, conn: ssl.SSLSocket):
        """

        This is the last procedure in PEFL protocol.
        The SP get a model encrypted by public key c, encrypt this model by public key x and
        then send back to SP.

        :param conn: the TCP connection that be used to make this procedure.
        :return: None.
        """

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        omega = [paillier.EncryptedNumber(self.skc.public_key, i)
                 for i in data]

        omega_x = arr_enc(arr_dec(omega, self.skc, self.precision), self.pkx, self.precision)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC,
            MessageItems.DATA: [i.ciphertext() for i in omega_x]
        }
        send_obj(conn, msg)
