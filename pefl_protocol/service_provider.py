"""This is the service provider part of the PEFL protocol

The class is inherit base service class and key requester class, so it is able to listen TCP
connection request and request key from the Key Generation Center.

Typical usage example:

cp = ServiceProvider(listening, cert_path, key_path, key_generator, cloud_provider, token_path,
                     model_length, model_range, learning_rate, trainers_count, train_round)
cp.run()

"""

from copy import deepcopy
import logging
from phe import paillier
from random import random
from pefl_protocol.base_service import BaseService
from pefl_protocol.configs import Configs
from pefl_protocol.connector import Connector
from pefl_protocol.consts import Protocols, MessageItems
from pefl_protocol.helpers import send_obj, receive_obj, make_logger
from pefl_protocol.enc_utils import Encryptor, gen_ciphertext
from pefl_protocol.key_generator import KeyRequester


class ServiceProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector, cloud_provider: Connector,
                 token_path: str, init_model: [float],
                 model_length: int, learning_rate: float,
                 trainers_count: int, train_round: int,
                 time_out=10, max_connection=5,
                 precision=32, value_range_bits=16,
                 logger: logging.Logger = None):

        if logger is None:
            self.logger = make_logger('ServiceProvider')
        else:
            self.logger = logger

        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection, logger=self.logger)
        KeyRequester.__init__(self, key_generator, token_path)

        self.cloud_provider = cloud_provider
        self.model_length = model_length
        self.trainers_count = trainers_count
        self.train_round = train_round
        self.learning_rate = learning_rate
        self.precision = precision
        self.value_bits = value_range_bits

        self.init_model = init_model
        self.model = []
        self.gradient = [[] for _ in range(self.trainers_count)]
        self.model_x = []
        self.is_ready = False

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

    def round_ready(self):
        """

        This function contact to every trainers, get their gradients and generate a
        id in this round for them.

        :return: None.
        """

        self.sock.listen(5)

        self.is_ready = False
        ready_list = []
        while len(ready_list) < self.trainers_count:
            conn, address = self.sock.accept()
            msg = receive_obj(conn)
            if msg[MessageItems.PROTOCOL] != Protocols.ROUND_READY \
                    or msg[MessageItems.USER] in ready_list:
                msg = {
                    MessageItems.PROTOCOL: msg[MessageItems.PROTOCOL],
                    MessageItems.DATA: 'Error'
                }
                send_obj(conn, msg)
                continue

            ready_list.append(msg[MessageItems.USER])

            msg = {
                MessageItems.PROTOCOL: Protocols.ROUND_READY,
                MessageItems.DATA: len(ready_list) - 1  # This is the id for this user.
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            data = msg[MessageItems.DATA]
            g = [paillier.EncryptedNumber(self.pkc, i) for i in data]

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

        r1 = [[self.gradient[i][j] + r[j] for j in range(enc_n)] for i in range(m)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED,
            MessageItems.DATA: {
                'r1': [gen_ciphertext(i) for i in r1],
                'm': m,
                'n': n,
                'enc_n': enc_n
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        dc = self.enc_c.gen_enc_number(data)
        gm = [dc[i] - r[i] for i in range(enc_n)]

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
        rx = [r0 * i for i in gx]
        ry = [r1 * i for i in gy]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER,
            MessageItems.DATA: {
                'rx': gen_ciphertext(rx),
                'ry': gen_ciphertext(ry),
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

        g1 = [[self.gradient[i][j] + r2[i][j] for j in range(enc_n)] for i in range(m)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG,
            MessageItems.DATA: {
                'nu': self.learning_rate,
                'g': [gen_ciphertext(i) for i in g1],
                'n': self.model_length
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        ex = [self.enc_c.gen_enc_number(i) for i in data['ex']]
        # "k" here is plain text!
        k = data['k']
        r1 = [k[i] * r[i] for i in range(m)]
        r2 = [self.enc_c.arr_enc([i] * n) for i in r1]
        fx = [[ex[i][j] - r2[i][j] for j in range(enc_n)] for i in range(m)]

        for i in range(m):
            for j in range(enc_n):
                self.model[j] = self.model[j] - fx[i][j]

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
        r1 = [self.model[i] + rc[i] for i in range(enc_n)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC,
            MessageItems.DATA: {
                'g': gen_ciphertext(r1),
                'n': n
            }
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        gx = self.enc_x.gen_enc_number(data)
        rc = self.enc_x.arr_enc(r)
        self.model_x = [gx[i] - rc[i] for i in range(enc_n)]

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
                MessageItems.DATA: [i.ciphertext() for i in self.model_x]
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            if msg[MessageItems.DATA] == 'OK':
                conn.close()
            else:
                self.logger.warning('Require model ended incorrectly.')
