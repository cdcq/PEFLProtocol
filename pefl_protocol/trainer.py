"""This model is for trainer machines (data owners)

The trainer class can contact to the service provider, send the gradient vector and
receive the next round model automatically. You just need to make a trainer object, then
send the gradient vector and receive the model every round.

Typical usage example:

t = trainer(key_generator, service_provider, token_path, model_length)
for i in range(round_count):
    gradient = train(model)
    model = t.round_run(gradient)

"""

import logging
from phe import paillier
from hashlib import md5
from random import getrandbits
from time import sleep

from pefl_protocol.connector import Connector
from pefl_protocol.consts import Protocols, MessageItems
from pefl_protocol.helpers import send_obj, receive_obj, arr_enc, arr_dec, make_logger
from pefl_protocol.key_generator import KeyRequester


class Trainer(KeyRequester):
    def __init__(self, key_generator: Connector, service_provider: Connector,
                 token_path: str,
                 model_length: int,
                 precision: int = 32,
                 wait_time: int = 5,
                 logger: logging.Logger = None):
        KeyRequester.__init__(self, key_generator, token_path)

        self.service_provider = service_provider
        self.model_length = model_length
        self.precision = precision
        self.wait_time = wait_time

        self.pkc = self.request_key(Protocols.GET_PKC)
        self.skx = self.request_key(Protocols.GET_SKX)

        self.user_name = md5(getrandbits(1024).to_bytes(1024, 'big')).hexdigest()
        self.round_id = -1
        self.gradient = []

        if logger is None:
            self.logger = make_logger('Trainer')
        else:
            self.logger = logger

    def round_run(self, gradient: [float]) -> [float]:
        self.logger.info('A new round is started. User name: {0}.'.format(self.user_name))
        self.round_id = -1
        self.gradient = gradient

        self.round_ready()
        return self.get_model()

    def round_ready(self):
        conn = self.service_provider.start_connect()
        msg = {
            MessageItems.PROTOCOL: Protocols.ROUND_READY,
            MessageItems.USER: self.user_name
        }
        send_obj(conn, msg)
        while True:
            try:
                msg = receive_obj(conn)
            except TimeoutError:
                sleep(2)
                continue

            break

        self.round_id = msg[MessageItems.DATA]

        gc = arr_enc(self.gradient, self.pkc, self.precision)
        msg = {
            MessageItems.PROTOCOL: Protocols.ROUND_READY,
            MessageItems.ID: self.round_id,
            MessageItems.DATA: [i.ciphertext() for i in gc]
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        if msg[MessageItems.DATA] == 'OK':
            conn.close()
            self.logger.info('The round is ready. The round ID is {}.'.format(self.round_id))
        else:
            self.logger.warning('Round {} ready ended incorrectly.'.format(self.round_id))

    def get_model(self) -> [float]:
        sleep(self.wait_time)
        while True:
            conn = self.service_provider.start_connect(wait_time=10)
            msg = {
                MessageItems.PROTOCOL: Protocols.GET_MODEL,
                MessageItems.USER: self.user_name
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            data = msg[MessageItems.DATA]
            if data != 'Error':
                break

        model = [paillier.EncryptedNumber(self.skx.public_key, data[i]) for i in range(len(data))]
        ret = arr_dec(model, self.skx, self.precision)

        msg = {
            MessageItems.PROTOCOL: Protocols.GET_MODEL,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

        return ret
