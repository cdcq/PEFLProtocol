import json
import socket
import ssl
from phe import paillier
from time import sleep


class Connector:
    # The length of an encrypted number encoded by encrypted2dict.
    LENGTH_OF_ENCRYPTED = 700

    def __init__(self, cloud: (str, int), time_out=10,
                 public_key: paillier.PaillierPublicKey = None,
                 context: ssl.SSLContext = None):
        self.cloud = cloud
        self.time_out = time_out
        self.public_key = public_key
        self.context = context

    def create_context(self, ca_path: str):
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_verify_locations(ca_path)

    def set_context(self, context: ssl.SSLContext):
        self.context = context

    def load_public_key(self, n: int):
        self.public_key = paillier.PaillierPublicKey(n)

    def start_connect(self) -> ssl.SSLSocket:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                conn.connect(self.cloud)
                break
            except ConnectionRefusedError:
                print('Connection has been refused, trying to reconnect.')
                sleep(2)
                continue

        return self.context.wrap_socket(conn)

    def get_public_key(self):
        sock = self.start_connect()

        msg = {
            'Protocol': 'GetPub'
        }
        sock.send(json.dumps(msg).encode())

        msg = sock.recv(self.LENGTH_OF_ENCRYPTED + 100)
        data = json.loads(msg)['Data']
        self.public_key = paillier.PaillierPublicKey(int(data))
