from phe import paillier
from hashlib import md5
from random import getrandbits
from time import sleep

from connector import Connector
from enums import PROTOCOLS
from helpers import send_obj, receive_obj, arr_enc
from key_generator import KeyRequester


class Trainer(KeyRequester):
    def __init__(self, key_generator: Connector, service_provider: Connector,
                 token_path: str,
                 model_length: int,
                 precision: int = 32):
        KeyRequester.__init__(self, key_generator, token_path)

        self.service_provider = service_provider
        self.model_length = model_length
        self.precision = precision

        self.pkc = self.request_key(PROTOCOLS.GET_PKC)
        self.skx = self.request_key(PROTOCOLS.GET_SKX)

        self.user_name = md5(getrandbits(1024)).hexdigest()
        self.round_id = -1
        self.gradient, _ = arr_enc([0 for _ in range(self.model_length)], self.pkc)

    def round_run(self, gradient: [float]):
        print('A new round is started. User name: {0}.'.format(self.user_name))
        self.round_id = -1
        self.gradient, _ = arr_enc(gradient, self.pkc, self.precision)

        self.round_ready()

    def round_ready(self):
        conn = self.service_provider.start_connect()
        msg = {
            'Protocol': PROTOCOLS.ROUND_READY,
            'User': self.user_name
        }
        send_obj(conn, msg)
        while True:
            try:
                msg = receive_obj(conn)
            except TimeoutError:
                sleep(2)
                continue

            break

        self.round_id = msg['Data']

        msg = {
            'Protocol': PROTOCOLS.ROUND_READY,
            'ID': self.round_id,
            'Data': [i.ciphertext() for i in self.gradient]
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        if msg['Data'] == 'OK':
            conn.close()
            print('The round is ready. Round ID: {0}.'.format(self.round_id))
        else:
            print('A round ready ended incorrectly. User name: {0}.'.format(self.user_name))
