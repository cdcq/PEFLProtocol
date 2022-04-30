import json
import numpy
from hashlib import md5
from random import getrandbits
from time import sleep

from connector import Connector
from enums import PROTOCOLS
from key_generator import KeyRequester


class Trainer(KeyRequester):
    def __init__(self, key_generator: Connector, service_provider: Connector,
                 token_path: str,
                 model_length: int,
                 exponent: int = 32):
        KeyRequester.__init__(self, key_generator, token_path)

        self.service_provider = service_provider
        self.model_length = model_length
        self.exponent = exponent

        self.pkc = self.request_key(PROTOCOLS.GET_PKC)
        self.skx = self.request_key(PROTOCOLS.GET_SKX)

        self.user_name = md5(getrandbits(1024)).hexdigest()
        self.round_id = -1
        self.gradient = [0 for _ in range(self.model_length)]

    def round_run(self, gradient: numpy.ndarray):
        print('A new round is started. User name: {0}.'.format(self.user_name))
        self.round_id = -1
        # TODO: ciphertext packing.
        self.gradient = [int(gradient[i] * (2 ** self.exponent))
                         for i in range(self.model_length)]

        self.round_ready()

    def round_ready(self):
        conn = self.service_provider.start_connect()
        msg = {
            'Protocol': PROTOCOLS.ROUND_READY,
            'User': self.user_name
        }
        conn.send(json.dumps(msg).encode())
        while True:
            try:
                msg = conn.recv(1024)
            except TimeoutError:
                sleep(2)
                continue

            break

        msg = json.loads(msg)
        self.round_id = msg['Data']

        msg = {
            'Protocol': PROTOCOLS.ROUND_READY,
            'ID': self.round_id,
            'Data': self.gradient
        }
        conn.send(json.dumps(msg).encode())

        msg = conn.recv(1024)
        msg = json.loads(msg)
        if msg['Data'] == 'OK':
            conn.close()
            print('The round is ready. Round ID: {0}.'.format(self.round_id))
        else:
            print('A round ready ended incorrectly. User name: {0}.'.format(self.user_name))
