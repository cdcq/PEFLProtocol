import json
import numpy
import socket
import ssl
import threading
from phe import paillier

from PHEHelper import encrypted2dict, dict2encrypted


class CloudProvider:
    # The length of an encrypted number encoded by encrypted2dict.
    LENGTH_OF_ENCRYPTED = 700

    def __init__(self, time_out=10, gradient_size=1000,
                 public_key: paillier.PaillierPublicKey = None,
                 private_key: paillier.PaillierPrivateKey = None,
                 sock: ssl.SSLSocket = None):
        self.time_out = time_out
        self.gradient_size = gradient_size
        self.public_key = public_key
        self.private_key = private_key
        self.sock = sock

    def create_socket(self, listening: (str, int), cert_path: str, key_path: str):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(listening)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        self.sock = context.wrap_socket(sock, server_side=True)

    def set_socket(self, sock: ssl.SSLSocket):
        self.sock = sock

    def load_private_key(self, p: int, q: int):
        self.public_key = paillier.PaillierPublicKey(p * q)
        self.private_key = paillier.PaillierPrivateKey(self.public_key, p, q)

    def run(self):
        if self.private_key is None:
            raise Exception('The cloud provider has no private key. :(')

        self.sock.listen(5)
        print('Waiting for connection.')
        while True:
            sock, address = self.sock.accept()
            t = threading.Thread(target=self.tcp_handler, args=(sock, address))
            t.start()

    def tcp_handler(self, sock: ssl.SSLSocket, address):
        sock.settimeout(self.time_out)
        print('Start a connection.')
        # TODO: receive data after the protocol has been confirmed.
        msg = sock.recv(self.gradient_size * self.LENGTH_OF_ENCRYPTED * 2 + 100)
        try:
            msg = json.loads(msg)
        except json.decoder.JSONDecodeError:
            sock.close()
            print('The data structure is wrong.')
            return
        if 'Protocol' not in msg:
            sock.close()
            print('The protocol is wrong.')
            return

        if msg['Protocol'] == 'GetPub':
            self.public_key_handler(sock)
        elif msg['Protocol'] == 'SecMed':
            self.medians_handler(msg, sock)
        elif msg['Protocol'] == 'SecPear':
            self.pearson_handler(msg, sock)

        sock.close()
        print('A connection has benn closed.')

    def public_key_handler(self, sock: ssl.SSLSocket):
        msg = {
            'Protocol': 'GetPub',
            'Data': self.public_key.n
        }
        sock.send(json.dumps(msg).encode())

    def medians_handler(self, msg, sock: ssl.SSLSocket):
        self.public_key: paillier.PaillierPublicKey
        self.private_key: paillier.PaillierPrivateKey

        r1 = msg['Data']
        m, n = len(r1), len(r1[0])

        r2 = [[dict2encrypted(r1[i][j], self.public_key)
               for i in range(m)] for j in range(n)]
        dt = [[self.private_key.decrypt(r2[j][i]) for i in range(n)] for j in range(m)]
        '''
        'dt' is a turned matrix.
        The above code is same as the following:
        d1 = [[self.private_key.decrypt(r2[i][j]) for i in range(m)] for j in range(n)]
        dt = [[d1[j][i] for i in range(n)] for j in range(m)]
        '''

        for i in range(n):
            dt[i].sort()

        if m % 2 == 1:
            dm = [dt[i][m // 2 + 1] for i in range(n)]
        else:
            dm = [(dt[i][m // 2] + dt[i][m // 2 + 1]) / 2 for i in range(n)]

        dc = [self.public_key.encrypt(dm[i]) for i in range(n)]
        msg = {
            'Protocol': 'SecMed',
            'Data': [encrypted2dict(dc[i]) for i in range(n)]
        }
        sock.send(json.dumps(msg).encode())

    def pearson_handler(self, msg, sock: ssl.SSLSocket):
        rx = msg['Data']['Rx']
        ry = msg['Data']['Ry']
        n = len(rx)

        dx = [self.private_key.decrypt(dict2encrypted(rx[i], self.public_key)) for i in range(n)]
        dy = [self.private_key.decrypt(dict2encrypted(ry[i], self.public_key)) for i in range(n)]

        rho = float(numpy.corrcoef(dx, dy)[0][1])

        msg = {
            'Protocol': 'SecPear',
            'Data': rho
        }
        sock.send(json.dumps(msg).encode())
