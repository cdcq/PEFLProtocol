"""This is connector model

The connector class inherit the function that be used when connect to a server.

Typical usage example:

sock = Connector(service, ca_path)
conn = sock.start_connect()
conn.send(data)
data = conn.recv()

"""

import logging
import socket
import ssl
from random import random
from time import sleep

from pefl_protocol.helpers import make_logger


class Connector:
    def __init__(self, service: (str, int), ca_path: str,
                 time_out=60,
                 logger: logging.Logger = None):
        self.service = service
        self.time_out = time_out

        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.context.load_verify_locations(ca_path)
        self.context.check_hostname = False

        if logger is None:
            self.logger = make_logger('Connector')
        else:
            self.logger = logger

    def start_connect(self, wait_time=4) -> ssl.SSLSocket:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                conn.connect(self.service)
                break
            except ConnectionRefusedError:
                self.logger.info('Connection has been refused. Trying to reconnect.')
                sleep(wait_time * (0.5 + 0.5 * random()))
                continue
            except TimeoutError:
                self.logger.info('Connection is timeout. Trying to reconnect.')
                sleep(wait_time * (0.5 + 0.5 * random()))
                continue

        return self.context.wrap_socket(conn)
