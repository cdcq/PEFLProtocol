"""This is the service provider part of the PEFL protocol

The class is inherit base service class and key requester class, so it is able to listen TCP
connection request and request key from the Key Generation Center.

Typical usage example:

cp = ServiceProvider(listening, cert_path, key_path, key_generator, cloud_provider, token_path,
                     model_length, model_range, learning_rate, trainers_count, train_round)
cp.run()

"""

from phe import paillier
from random import random, getrandbits

from base_service import BaseService
from connector import Connector
from consts import Protocols, MessageItems
from helpers import send_obj, receive_obj, arr_enc, arr_enc_len
from key_generator import KeyRequester


class ServiceProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector, cloud_provider: Connector,
                 token_path: str,
                 model_length: int, learning_rate: int,
                 trainers_count: int, train_round: int,
                 time_out=10, max_connection=5):
        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection)
        KeyRequester.__init__(self, key_generator, token_path)

        self.cloud_provider = cloud_provider
        self.model_length = model_length
        self.trainers_count = trainers_count
        self.train_round = train_round
        self.learning_rate = learning_rate

        self.model = []
        self.gradient = [[] for _ in range(self.trainers_count)]
        self.r0 = []
        self.model_x = []

        self.pkc = self.request_key(Protocols.GET_PKC)

    def run(self):
        # self.model = self.init_model()
        self.model = arr_enc(self.model, self.pkc)

        for round_number in range(1, self.train_round + 1):
            self.round(round_number)

    def round(self, round_number):
        print('Round {0} is started, waiting for trainers ready.'.format(round_number))
        self.round_ready()
        print('All trainers are ready.')
        self.cloud_init()

        print('Start SEC_MED.')
        gm = self.medians_protocol()
        print('Start SEC_PER.')
        for i in range(self.trainers_count):
            self.pearson_protocol(self.gradient[i], gm, i)

        print('Start SEC_AGG.')
        self.aggregate_protocol()
        print('Start SEC_EXC.')
        self.exchange_protocol()

        print('All protocols are finished. Start to distribute model.')
        self.distribute_model()
        print('A round finished.')

    def round_ready(self):
        """

        This function contact to every trainers, get their gradients and generate a
        id in this round for them.

        :return: None.
        """

        self.sock.listen(1)

        ready_list = []
        while len(ready_list) < self.trainers_count:
            conn, address = self.sock.accept()
            msg = receive_obj(conn)
            if msg[MessageItems.PROTOCOL] != Protocols.ROUND_READY \
                    or msg[MessageItems.USER] in ready_list:
                conn.close()
                continue

            ready_list.append(msg[MessageItems.USER])

            msg = {
                MessageItems.PROTOCOL: Protocols.ROUND_READY,
                MessageItems.DATA: len(ready_list)  # This is the id for this user.
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            data = msg[MessageItems.DATA]
            g = [paillier.EncryptedNumber(self.pkc, data[i]) for i in range(len(data))]

            self.gradient[msg[MessageItems.ID]] = g

            msg = {
                MessageItems.PROTOCOL: Protocols.ROUND_READY,
                MessageItems.DATA: 'OK'
            }
            send_obj(conn, msg)

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

        m, n = self.trainers_count, arr_enc_len(self.model_length)
        # Attention, if this number will overflow after plus?
        r = [self.pkc.encrypt(getrandbits(2048)) for _ in range(n)]
        self.r0 = r

        r1 = [[(self.gradient[i][j] + r[j]).ciphertext() for j in range(n)] for i in range(m)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_MED,
            MessageItems.DATA: r1
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        dc = [paillier.EncryptedNumber(self.pkc, data[i]) for i in range(n)]
        gm = [dc[i] - r[i] for i in range(n)]

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

        n = arr_enc_len(self.model_length)
        r1 = [self.pkc.encrypt(getrandbits(32)) for _ in range(n)]
        r2 = [self.pkc.encrypt(getrandbits(32)) for _ in range(n)]
        rx = [r1[i] * gx[i] for i in range(n)]
        ry = [r2[i] * gy[i] for i in range(n)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_PER,
            MessageItems.DATA: {
                'rx': [rx[i].ciphertext() for i in range(n)],
                'ry': [ry[i].ciphertext() for i in range(n)],
                'x_id': x_id
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

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_AGG,
            MessageItems.DATA: self.learning_rate
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        m, n = self.trainers_count, arr_enc_len(self.model_length)
        ex = [[paillier.EncryptedNumber(self.pkc, data['ex'][i][j])
               for j in range(n)] for i in range(m)]
        k = [paillier.EncryptedNumber(self.pkc, data['k'][i]) for i in range(n)]
        fx = [[ex[i][j] - k[i] * self.r0[i] for j in range(n)] for i in range(m)]

        for i in range(m):
            for j in range(n):
                self.model[j] = self.model[j] + fx[i][j]

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

        n = arr_enc_len(self.model_length)
        r = [self.pkc.encrypt(getrandbits(2048)) for _ in range(n)]
        r1 = [self.model[i] + r[i] for i in range(n)]

        msg = {
            MessageItems.PROTOCOL: Protocols.SEC_EXC,
            MessageItems.DATA: [r1[i].ciphertext() for i in range(n)]
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        self.model_x = msg[MessageItems.DATA]

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

        self.sock.listen(1)

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
                MessageItems.DATA: self.model_x
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            if msg[MessageItems.DATA] == 'OK':
                conn.close()
