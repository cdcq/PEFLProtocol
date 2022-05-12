"""This is connector model

The connector class inherit the function that be used when connect to a server.

Typical usage example:

sock = Connector(service, ca_path)
conn = sock.start_connect()
conn.send(data)
data = conn.recv()

"""

import socket
import ssl
from time import sleep


class Connector:
    def __init__(self, service: (str, int), ca_path: str,
                 time_out=10):
        self.service = service
        self.time_out = time_out

        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.context.load_verify_locations(ca_path)
        self.context.check_hostname = False

    def start_connect(self) -> ssl.SSLSocket:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                conn.connect(self.service)
                break
            except ConnectionRefusedError:
                print('Connection has been refused, trying to reconnect.')
                sleep(2)
                continue

        return self.context.wrap_socket(conn)
