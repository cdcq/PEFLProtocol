"""This is the service provider part of the PEFL protocol

The class is inherit base service class and key requester class, so it is able to listen TCP
connection request and request key from the Key Generation Center.

Typical usage example:

cp = ServiceProvider(listening, cert_path, key_path, key_generator, cloud_provider, token_path,
                     model_length, model_range, learning_rate, trainers_count, train_round)
cp.run()

"""

import logging
from phe import paillier
from random import random
from time import time

from pefl_protocol.base_service import BaseService
from pefl_protocol.configs import Configs
from pefl_protocol.connector import Connector
from pefl_protocol.consts import Protocols, MessageItems
from pefl_protocol.helpers import send_obj, receive_obj, make_logger
from pefl_protocol.enc_utils import Encryptor, gen_cipher_arr
from pefl_protocol.key_generator import KeyRequester
from pefl_protocol.multiprocess_utils import arr_add, arr_sub, arr_mul


class ServiceProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector, cloud_provider: Connector,
                 token_path: str, init_model: [float],
                 model_length: int, learning_rate: float,
                 trainers_count: int, train_round: int,
                 time_out=10, max_connection=5,
                 precision=32, value_range_bits=16,
                 logger: logging.Logger = None,
                 mu_export_path: str = None):

        if logger is None:
            self.logger = make_logger('ServiceProvider')
        else:
            self.logger = logger

        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection, logger=self.logger)
        KeyRequester.__init__(self, key_generator, token_path)
        self.key_generator.logger = self.logger

        self.cloud_provider = cloud_provider

        # TODO: use a more appropriate way to set logger.
        self.cloud_provider.logger = self.logger

        self.model_length = model_length
        self.trainers_count = trainers_count
        self.train_round = train_round
        self.learning_rate = learning_rate
        self.precision = precision
        self.value_bits = value_range_bits
        self.mu_export_path = mu_export_path

        self.init_model = init_model
        self.model = []
        self.gradient = [[] for _ in range(self.trainers_count)]
        self.model_x = []
        self.is_ready = False
        self.round_number = 0
        self.user_list = []
        self.round_user = []
        self.mu_table = []

        self.pkc = self.request_key(Protocols.GET_PKC)
        self.pkx = self.request_key(Protocols.GET_PKX)

        # Used for debug.
        self.temp = []

        self.enc_c = Encryptor(self.pkc, None, self.precision, self.value_bits,
                               Configs.KEY_LENGTH, Configs.IF_PACKAGE)
        self.enc_x = Encryptor(self.pkx, None, self.precision, self.value_bits,
                               Configs.KEY_LENGTH, Configs.IF_PACKAGE)

    def run(self):
        self.model = self.enc_c.arr_enc(self.init_model)

        for round_number in range(1, self.train_round + 1):
            self.round(round_number)

    def round(self, round_number):
        self.round_number = round_number
        self.logger.info('Round {0} is started, waiting for trainers ready.'.format(round_number))

        self.round_ready()
        self.logger.info('All trainers are ready.')

        self.cloud_init()

        self.logger.info('Start SecMed.')
        gm = self.medians_protocol()

        self.logger.info('Start SecPear.')
        for i in range(self.trainers_count):
            self.pearson_protocol(self.gradient[i], gm, i)

        self.logger.info('Start SecAgg.')
        self.aggregate_protocol()

        self.logger.info('Start SecExch.')
        self.exchange_protocol()

        self.logger.info('Start to distribute model.')
        self.distribute_model()

        self.export_mu_table()

    def round_ready(self):
        """

        This function contact to every trainers, get their gradients and generate a
        id in this round for them.

        :return: None.
        """

        self.sock.listen(10)

        self.is_ready = False
        self.round_user = []
        while len(self.round_user) < self.trainers_count:
            conn, address = self.sock.accept()
            msg = receive_obj(conn)
            if msg[MessageItems.PROTOCOL] != Protocols.ROUND_READY \
                    or msg[MessageItems.USER] in self.round_user:
                msg = {
                    MessageItems.PROTOCOL: msg[MessageItems.PROTOCOL],
                    MessageItems.DATA: 'Error'
                }
                send_obj(conn, msg)
                continue

            self.round_user.append(msg[MessageItems.USER])

            if msg[MessageItems.USER] not in self.user_list:
                self.user_list.append(msg[MessageItems.USER])

            msg = {
                MessageItems.PROTOCOL: Protocols.ROUND_READY,
                MessageItems.DATA: len(self.round_user) - 1  # This is the id for this user.
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            data = msg[MessageItems.DATA]
            g = self.enc_c.gen_encrypted_arr(data)

            self.gradient[msg[MessageItems.ID]] = g

            msg = {
                MessageItems.PROTOCOL: Protocols.ROUND_READY,
                MessageItems.DATA: 'OK'
            }
            send_obj(conn, msg)

        self.is_ready = True

    def cloud_init(self):
        """

        This function make some initialization of the cloud provider before a train
        round start.

        :return: None.
        """

        conn = self.cloud_provider.start_connect()
        msg = {
            MessageItems.PROTOCOL: Protocols.CLOUD_INIT
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        msg = {
            MessageItems.PROTOCOL: Protocols.CLOUD_INIT,
            MessageItems.DATA: {
                'trainers_count': self.trainers_count
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        msg = {
            MessageItems.PROTOCOL: Protocols.CLOUD_INIT,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

    def medians_protocol(self) -> [paillier.EncryptedNumber]:
        """

        This function realize the "SecMed" procedure in the PEFL protocol.

        :return: the medians vector or gradients.
        """

        conn = self.cloud_provider.start_connect()
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        m, n = self.trainers_count, self.model_length
        enc_n = self.enc_c.arr_enc_len(self.model_length)
        # TODO: is random suitable?
        r = [random() for _ in range(self.model_length)]
        r = self.enc_c.arr_enc(r)

        # r1 = [[self.gradient[i][j] + r[j] for j in range(enc_n)] for i in range(m)]
        r1 = [arr_add(i, r) for i in self.gradient]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED,
            MessageItems.DATA: {
                'r1': [gen_cipher_arr(i) for i in r1],
                'm': m,
                'n': n,
                'enc_n': enc_n
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        dc = self.enc_c.gen_encrypted_arr(data)
        # gm = [dc[i] - r[i] for i in range(enc_n)]
        gm = arr_sub(dc, r)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

        return gm

    def pearson_protocol(self, gx: [paillier.EncryptedNumber], gy: [paillier.EncryptedNumber],
                         x_id: int):
        """

        This function realize the "SecPear" procedure in the PEFL protocol.

        :param gx: the gradient vector of a trainer.
        :param gy: the medians vector.
        :param x_id: the round id of the trainer.
        :return: None.
        """

        conn = self.cloud_provider.start_connect()
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        r0 = int(random() * 2 ** self.precision)
        r1 = int(random() * 2 ** self.precision)
        # rx = [r0 * i for i in gx]
        # ry = [r1 * i for i in gy]
        rx = arr_mul(gx, [r0] * len(gx))
        ry = arr_mul(gy, [r1] * len(gy))

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER,
            MessageItems.DATA: {
                'rx': gen_cipher_arr(rx),
                'ry': gen_cipher_arr(ry),
                'x_id': x_id,
                'n': self.model_length
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

    def aggregate_protocol(self):
        """

        This function realize the "SecAgg" procedure in the PEFL protocol.

        :return: None.
        """

        conn = self.cloud_provider.start_connect()
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        m, n = self.trainers_count, self.model_length
        enc_n = self.enc_c.arr_enc_len(self.model_length)

        r = [random() for _ in range(m)]
        r2 = [self.enc_c.arr_enc([i] * n) for i in r]

        # g1 = [[self.gradient[i][j] + r2[i][j] for j in range(enc_n)] for i in range(m)]
        g1 = [arr_add(self.gradient[i], r2[i]) for i in range(m)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG,
            MessageItems.DATA: {
                'nu': self.learning_rate,
                'g': [gen_cipher_arr(i) for i in g1],
                'n': self.model_length
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        ex = [self.enc_c.gen_encrypted_arr(i) for i in data['ex']]
        # "k" here is plain text!
        k = data['k']
        r1 = [k[i] * r[i] for i in range(m)]
        r2 = [self.enc_c.arr_enc([i] * n) for i in r1]
        # fx = [[ex[i][j] - r2[i][j] for j in range(enc_n)] for i in range(m)]
        fx = [arr_sub(ex[i], r2[i]) for i in range(m)]

        for i in range(m):
            # for j in range(enc_n):
            #     self.model[j] = self.model[j] - fx[i][j]
            self.model = arr_sub(self.model, fx[i])

        mu = data['mu']
        mu_sorted = [0] * self.trainers_count
        for i in range(self.trainers_count):
            mu_sorted[self.user_list.index(self.round_user[i])] = mu[i]
        self.mu_table.append(mu_sorted)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

    def exchange_protocol(self):
        """

        This function realize the "SecExch" procedure in the PEFL protocol.

        :return: the next round model encrypted by public key x.
        """

        conn = self.cloud_provider.start_connect()
        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        n = self.model_length
        enc_n = self.enc_c.arr_enc_len(n)
        r = [random() for _ in range(n)]
        rc = self.enc_c.arr_enc(r)
        # r1 = [self.model[i] + rc[i] for i in range(enc_n)]
        r1 = arr_add(self.model, rc)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC,
            MessageItems.DATA: {
                'g': gen_cipher_arr(r1),
                'n': n
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        gx = self.enc_x.gen_encrypted_arr(data)
        rc = self.enc_x.arr_enc(r)
        # self.model_x = [gx[i] - rc[i] for i in range(enc_n)]
        self.model_x = arr_sub(gx, rc)

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

    def distribute_model(self):
        """

        This function distribute the next round model to every trainer.

        :return: None.
        """

        self.sock.listen(5)

        distribute_list = []
        while len(distribute_list) < self.trainers_count:
            conn, address = self.sock.accept()
            msg = receive_obj(conn)
            if msg[MessageItems.PROTOCOL] != Protocols.GET_MODEL \
                    or msg[MessageItems.USER] in distribute_list:
                conn.close()
                continue

            distribute_list.append(msg[MessageItems.USER])

            msg = {
                MessageItems.PROTOCOL: Protocols.GET_MODEL,
                MessageItems.DATA: gen_cipher_arr(self.model_x)
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            if msg[MessageItems.DATA] == 'OK':
                conn.close()
            else:
                self.logger.warning('Require model ended incorrectly.')

    def export_mu_table(self):
        if self.mu_export_path is None:
            return
        with open(self.mu_export_path, 'w') as f:
            f.write('round, ' + ', '.join(self.user_list) + '\n')
            for i in range(len(self.mu_table)):
                f.write(str(i) + ', ' + ', '.join([str(j) for j in self.mu_table[i]]) + '\n')
