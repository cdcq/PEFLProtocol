from phe import paillier
from random import random, getrandbits

from base_service import BaseService
from connector import Connector
from enums import PROTOCOLS
from helpers import send_obj, receive_obj
from key_generator import KeyRequester


class ServiceProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector, cloud_provider: Connector,
                 token_path: str,
                 model_length: int, model_range: (int, int),
                 trainers_count: int, train_round: int,
                 time_out=10, max_connection=5):
        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection)
        KeyRequester.__init__(self, key_generator, token_path)

        self.cloud_provider = cloud_provider
        self.model_length = model_length
        self.trainers_count = trainers_count
        self.train_round = train_round
        self.model_range = model_range

        self.model = [random() * (model_range[1] - model_range[0]) + model_range[0]
                      for _ in range(self.model_length)]
        self.gradient = [[0 for _ in range(self.model_length)]
                         for _ in range(self.trainers_count)]

        self.pkc = self.request_key(PROTOCOLS.GET_PKC)

    def run(self):
        for round_number in range(1, self.train_round + 1):
            self.round(round_number)

    def round(self, round_number):
        print('Round {0} is started, waiting for trainers ready.'.format(round_number))
        self.round_ready()
        print('All trainers are ready.')

    def round_ready(self):
        self.sock.listen(1)

        ready_list = []
        while len(ready_list) < self.trainers_count:
            conn, address = self.sock.accept()
            msg = receive_obj(conn)
            if msg['Protocol'] != PROTOCOLS.ROUND_READY \
                    or msg['User'] in ready_list:
                conn.close()
                continue

            ready_list.append(msg['User'])

            msg = {
                'Protocol': PROTOCOLS.ROUND_READY,
                'Data': len(ready_list)  # This is the id for this user.
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)

            self.gradient[msg['ID']] = msg['Data']

            send_obj(conn, msg)

    def medians_protocol(self) -> [paillier.EncryptedNumber]:
        conn = self.cloud_provider.start_connect()
        msg = {
            'Protocol': PROTOCOLS.SEC_MED
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)

        m, n = self.trainers_count, self.model_length
        # Attention, if this number will overflow after plus?
        r = [self.pkc.encrypt(getrandbits(2048)) for _ in range(n)]
        r1 = [[(self.gradient[i][j] + r[j]).ciphertext() for j in range(n)] for i in range(m)]

        msg = {
            'Protocol': PROTOCOLS.SEC_MED,
            'Data': r1
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        dc = [paillier.EncryptedNumber(self.pkc, msg['Data'][i]) for i in range(n)]
        gm = [dc[i] - r[i] for i in range(n)]

        msg = {
            'Protocol': PROTOCOLS.SEC_MED,
            'Data': 'OK'
        }
        send_obj(conn, msg)

        return gm
