import socket
import ssl
from time import sleep


class Connector:
    def __init__(self, service: (str, int), ca_path: str,
                 time_out=10):
        self.service = service
        self.time_out = time_out

        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_verify_locations(ca_path)

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
