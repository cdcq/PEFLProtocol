import ssl
from phe import paillier

from base_service import BaseService
from connector import Connector
from enums import PROTOCOLS
from helpers import send_obj, receive_obj, arr_enc, arr_dec
from key_generator import KeyRequester


class CloudProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector,
                 token_path: str,
                 time_out=10, max_connection=5,
                 precision=32):
        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection)
        KeyRequester.__init__(self, key_generator, token_path)

        self.precision = precision

        self.skc = self.request_key(PROTOCOLS.GET_SKC)
        self.pkx = self.request_key(PROTOCOLS.GET_PKX)

    def tcp_handler(self, conn: ssl.SSLSocket, address):
        conn.settimeout(self.time_out)
        print('Start a connection.')
        msg = receive_obj(conn)

        protocol = msg['Protocol']
        if protocol == PROTOCOLS.SEC_MED:
            self.medians_handler(conn)

        msg = receive_obj(conn)
        if msg['Data'] == 'OK':
            conn.close()
            print('Closed a connection.')
        else:
            print('A protocol ended incorrectly. Protocol ID: {0}.'.format(protocol))

    def medians_handler(self, conn: ssl.SSLSocket):

        msg = {
            'Protocol': PROTOCOLS.SEC_MED
        }
        send_obj(conn, msg)

        msg = receive_obj(conn)
        r1 = msg['Data']
        m, n = len(r1), len(r1[0])

        gc = [[paillier.EncryptedNumber(self.skc.public_key, r1[i][j])
               for j in range(n)] for i in range(m)]
        gm = [arr_dec(gc[i], self.skc, self.precision)[0] for i in range(m)]
        n = len(gm[0])  # Attention.
        dt = [[gm[j][i] for j in range(m)] for i in range(n)]

        for i in range(n):
            dt[i].sort()

        if m % 2 == 1:
            dm = [dt[i][m // 2 + 1] for i in range(n)]
        else:
            dm = [(dt[i][m // 2] + dt[i][m // 2 + 1]) / 2 for i in range(n)]

        dc, n = arr_enc(dm, self.skc.public_key, self.precision)
        msg = {
            'Protocol': PROTOCOLS.SEC_MED,
            'Data': [dc[i].ciphertext() for i in range(n)]
        }
        send_obj(conn, msg)
