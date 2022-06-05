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

import logging
import math
import numpy
import ssl
from time import time

from pefl_protocol.base_service import BaseService
from pefl_protocol.configs import Configs
from pefl_protocol.connector import Connector
from pefl_protocol.consts import Protocols, MessageItems
from pefl_protocol.helpers import send_obj, receive_obj, make_logger
from pefl_protocol.enc_utils import Encryptor, gen_cipher_arr
from pefl_protocol.key_generator import KeyRequester


class CloudProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector,
                 token_path: str,
                 time_out=10, max_connection=5,
                 precision=32, value_range_bits=16,
                 logger: logging.Logger = None):
        if logger is None:
            self.logger = make_logger('CloudProvider')
        else:
            self.logger = logger

        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection, logger=self.logger)
        KeyRequester.__init__(self, key_generator, token_path)

        self.precision = precision
        self.value_bits = value_range_bits

        self.skc = self.request_key(Protocols.GET_SKC)
        self.pkx = self.request_key(Protocols.GET_PKX)
        self.trainers_count = 0
        self.mu = []

        self.enc_c = Encryptor(self.skc.public_key, self.skc, self.precision, self.value_bits,
                               Configs.KEY_LENGTH, Configs.IF_PACKAGE)
        self.enc_x = Encryptor(self.pkx, None, self.precision, self.value_bits,
                               Configs.KEY_LENGTH, Configs.IF_PACKAGE)

    def tcp_handler(self, conn: ssl.SSLSocket, address):
        """

        This function rewrite the function in super class, analysis the protocol of the
        TCP request and chose the correct processing function.

        :param conn: the TCP connection to be handled.
        :param address: no use.
        :return: None.
        """

        conn.settimeout(self.time_out)
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
        else:
            self.logger.warning('A {} protocol ended incorrectly.'.format(protocol))

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
        data = msg[MessageItems.DATA]
        r1 = data['r1']
        m, n, enc_n = data['m'], data['n'], data['enc_n']

        dc = [self.enc_c.gen_encrypted_arr(i) for i in r1]
        dx = [self.enc_c.arr_dec(i, n) for i in dc]

        # Transpose the dx matrix. It is convenient for sorting the column vectors in
        # matrix dx and then calculate the medians in a column vector.
        dt = [[dx[j][i] for j in range(m)] for i in range(n)]

        for i in range(n):
            dt[i].sort()

        if m % 2 == 1:
            dm = [dt[i][m // 2] for i in range(n)]
        else:
            dm = [(dt[i][m // 2 - 1] + dt[i][m // 2]) / 2 for i in range(n)]

        dc = self.enc_c.arr_enc(dm)
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED,
            MessageItems.DATA: gen_cipher_arr(dc)
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
        rx, ry, x_id, n = data['rx'], data['ry'], data['x_id'], data['n']
        rx = self.enc_c.gen_encrypted_arr(rx)
        ry = self.enc_c.gen_encrypted_arr(ry)

        dx = self.enc_c.arr_dec(rx, n)
        dy = self.enc_c.arr_dec(ry, n)

        rho = float(numpy.corrcoef(dx, dy)[0][1])

        small_number = 1e-10
        reject_number = 0.6
        # This is the weight formula mentioned in PEFL paper.
        self.mu[x_id] = max(0.0, math.log((1 + rho) / (1 - rho + small_number)) - reject_number)

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
        data = msg[MessageItems.DATA]
        nu, g, n = data['nu'], data['g'], data['n']

        g = [self.enc_c.gen_encrypted_arr(i) for i in g]
        gm = [self.enc_c.arr_dec(i, n) for i in g]
        m = self.trainers_count

        sum_mu = sum(self.mu)

        # "k" is the aggregate formula mentioned in PEFL paper.
        small_number = 1e-10
        k = [nu * self.mu[i] / (m * sum_mu + small_number) for i in range(m)]
        ex = [[k[i] * gm[i][j] for j in range(n)] for i in range(m)]

        ec = [self.enc_c.arr_enc(i) for i in ex]
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG,
            MessageItems.DATA: {
                'ex': [gen_cipher_arr(i) for i in ec],
                # "k" here is plain text!
                'k': k
            }
        }
        self.logger.info('SecAgg: the mu is\n{}'.format(self.mu))
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
        g, n = data['g'], data['n']
        g = self.enc_c.gen_encrypted_arr(g)

        gm = self.enc_c.arr_dec(g, n)
        gx = self.enc_x.arr_enc(gm)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC,
            MessageItems.DATA: gen_cipher_arr(gx)
        }
        send_obj(conn, msg)
