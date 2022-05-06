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

from phe import paillier
from hashlib import md5
from random import getrandbits
from time import sleep

from connector import Connector
from enums import Protocols, MessageItems
from helpers import send_obj, receive_obj, arr_enc, arr_dec
from key_generator import KeyRequester


class Trainer(KeyRequester):
    def __init__(self, key_generator: Connector, service_provider: Connector,
                 token_path: str,
                 model_length: int,
                 precision: int = 32,
                 wait_time: int = 5):
        KeyRequester.__init__(self, key_generator, token_path)

        self.service_provider = service_provider
        self.model_length = model_length
        self.precision = precision
        self.wait_time = wait_time

        self.pkc = self.request_key(Protocols.GET_PKC)
        self.skx = self.request_key(Protocols.GET_SKX)

        self.user_name = md5(getrandbits(1024)).hexdigest()
        self.round_id = -1
        self.gradient = []

    def round_run(self, gradient: [float]) -> [float]:
        print('A new round is started. User name: {0}.'.format(self.user_name))
        self.round_id = -1
        self.gradient = arr_enc(gradient, self.pkc, self.precision)

        self.round_ready()
        return self.get_mode()

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

        msg = {
            MessageItems.PROTOCOL: Protocols.ROUND_READY,
            MessageItems.ID: self.round_id,
            MessageItems.DATA: [i.ciphertext() for i in self.gradient]
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        if msg[MessageItems.DATA] == 'OK':
            conn.close()
            print('The round is ready. Round ID: {0}.'.format(self.round_id))
        else:
            print('A round ready ended incorrectly. User name: {0}.'.format(self.user_name))

    def get_mode(self) -> [float]:
        sleep(self.wait_time)
        conn = self.service_provider.start_connect()
        msg = {
            MessageItems.PROTOCOL: Protocols.GET_MODEL,
            MessageItems.USER: self.user_name
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        data = msg[MessageItems.DATA]
        model = [paillier.EncryptedNumber(self.skx.public_key, data[i]) for i in range(len(data))]
        ret = arr_dec(model, self.skx)

        msg = {
            MessageItems.PROTOCOL: Protocols.GET_MODEL,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)

        return ret
