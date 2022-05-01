import ssl
import yaml
from phe import paillier

from base_service import BaseService
from connector import Connector
from enums import PROTOCOLS
from helpers import send_obj, receive_obj


class KeyGenerator(BaseService):

    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 users_path: str,
                 time_out=10, max_connection=5):
        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection)

        # Very slow operation !!!
        self.pkx, self.skx = paillier.generate_paillier_keypair()
        self.pkc, self.skc = paillier.generate_paillier_keypair()

        self.users_path = users_path

    def tcp_handler(self, conn: ssl.SSLSocket, address):
        self.pkx: paillier.PaillierPublicKey
        self.skx: paillier.PaillierPrivateKey
        self.pkc: paillier.PaillierPublicKey
        self.skc: paillier.PaillierPrivateKey

        print('Start a connection.')
        conn.settimeout(self.time_out)

        msg = receive_obj(conn)

        with open(self.users_path, 'r') as f:
            users_data = f.read()
        users_list = yaml.safe_load(users_data)
        if msg['User'] not in users_list or users_list[msg['User']]['Token'] != msg['Token']:
            conn.close()
            print('The user info is wrong.')
            return

        user_right = users_list[msg['User']]['Right']
        # 01b is available for skx and 10b is available for skc.

        protocol = msg['Protocol']

        if protocol == PROTOCOLS.GET_PKX or protocol == PROTOCOLS.GET_PKC:
            send_key(conn, protocol, self.pkx.n)
        elif protocol == PROTOCOLS.GET_SKX and (user_right & 1) > 0:
            send_key(conn, protocol, (self.skx.p, self.skx.q))
        elif protocol == PROTOCOLS.GET_SKC and (user_right & 2) > 0:
            send_key(conn, protocol, (self.skc.p, self.skc.q))

        msg = receive_obj(conn)
        if msg['Data'] == 'OK':
            conn.close()
            print('Closed a connection.')
        else:
            print('A protocol ended incorrectly. Protocol ID: {0}.'.format(protocol))


def send_key(conn: ssl.SSLSocket, protocol, key):
    msg = {
        'Protocol': protocol,
        'Data': key
    }
    send_obj(conn, msg)


class KeyRequester:
    def __init__(self, key_generator: Connector, token_path):
        self.key_generator = key_generator
        self.token_path = token_path

    def request_key(self, protocol) \
            -> paillier.PaillierPublicKey | paillier.PaillierPrivateKey:
        with open(self.token_path, 'r') as f:
            token_data = f.read()
        token_data = yaml.safe_load(token_data)

        msg = {
            'Protocol': protocol,
            'User': token_data['User'],
            'Token': token_data['Token']
        }

        conn = self.key_generator.start_connect()
        send_obj(conn, msg)

        msg = receive_obj(conn)

        if protocol == PROTOCOLS.GET_PKC or protocol == PROTOCOLS.GET_PKX:
            ret = paillier.PaillierPublicKey(msg['Data'])
        elif protocol == PROTOCOLS.GET_SKC or protocol == PROTOCOLS.GET_SKX:
            pub = paillier.PaillierPublicKey(msg['Data'][0] * msg['Data'][1])
            ret = paillier.PaillierPrivateKey(pub, msg['Data'][0], msg['Data'][1])
        else:
            ret = None

        msg = {
            'Protocol': protocol,
            'Data': 'OK'
        }
        send_obj(conn, msg)
        return ret
