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
import yaml
from phe import paillier
from hashlib import md5
from random import getrandbits
from time import sleep

from pefl_protocol.configs import Configs
from pefl_protocol.connector import Connector
from pefl_protocol.consts import Protocols, MessageItems
from pefl_protocol.helpers import send_obj, receive_obj, make_logger
from pefl_protocol.enc_utils import Encryptor, gen_cipher_arr
from pefl_protocol.key_generator import KeyRequester


class Trainer(KeyRequester):
    def __init__(self, key_generator: Connector, service_provider: Connector,
                 token_path: str,
                 model_length: int,
                 precision=32, value_range_bits=16,
                 wait_time=10,
                 logger: logging.Logger = None):
        KeyRequester.__init__(self, key_generator, token_path)

        self.service_provider = service_provider
        # TODO: improve the authentication system.
        self.token_path = token_path
        self.model_length = model_length
        self.precision = precision
        self.value_bits = value_range_bits
        self.wait_time = wait_time

        self.pkc = self.request_key(Protocols.GET_PKC)
        self.skx = self.request_key(Protocols.GET_SKX)

        self.round_id = -1
        self.gradient = []

        if logger is None:
            self.logger = make_logger('Trainer')
        else:
            self.logger = logger

        self.service_provider.logger = self.logger
        self.key_generator.logger = self.logger

        self.enc_c = Encryptor(self.pkc, None, self.precision, self.value_bits,
                               Configs.KEY_LENGTH, Configs.IF_PACKAGE)
        self.enc_x = Encryptor(self.skx.public_key, self.skx, self.precision, self.value_bits,
                               Configs.KEY_LENGTH, Configs.IF_PACKAGE)

        with open(self.token_path, 'r') as f:
            token_data = f.read()
        token_data = yaml.safe_load(token_data)
        self.user_name = token_data['User']

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

        gc = self.enc_c.arr_enc(self.gradient)
        msg = {
            MessageItems.PROTOCOL: Protocols.ROUND_READY,
            MessageItems.ID: self.round_id,
            MessageItems.DATA: gen_cipher_arr(gc)
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
            conn = self.service_provider.start_connect(wait_time=self.wait_time)
            msg = {
                MessageItems.PROTOCOL: Protocols.GET_MODEL,
                MessageItems.USER: self.user_name
            }
            send_obj(conn, msg)

            msg = receive_obj(conn)
            data = msg[MessageItems.DATA]
            if data != 'Error':
                break

        model = self.enc_x.gen_encrypted_arr(data)
        ret = self.enc_x.arr_dec(model, self.model_length)

        msg = {
            MessageItems.PROTOCOL: Protocols.GET_MODEL,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

        return ret
