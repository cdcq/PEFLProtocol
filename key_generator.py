import ssl
import yaml
from phe import paillier

from base_service import BaseService
from connector import Connector
from enums import Protocols, MessageItems
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
        user_id = msg[MessageItems.USER]
        if user_id not in users_list \
                or users_list[user_id]['Token'] != msg[MessageItems.TOKEN]:
            conn.close()
            print('The user info is wrong.')
            return

        user_right = users_list[user_id]['Right']
        # 01b is available for skx and 10b is available for skc.

        protocol = msg[MessageItems.PROTOCOL]

        if protocol == Protocols.GET_PKX or protocol == Protocols.GET_PKC:
            send_key(conn, protocol, self.pkx.n)
        elif protocol == Protocols.GET_SKX and (user_right & 1) > 0:
            send_key(conn, protocol, (self.skx.p, self.skx.q))
        elif protocol == Protocols.GET_SKC and (user_right & 2) > 0:
            send_key(conn, protocol, (self.skc.p, self.skc.q))

        msg = receive_obj(conn)
        if msg[MessageItems.DATA] == 'OK':
            conn.close()
            print('Closed a connection.')
        else:
            print('A protocol ended incorrectly. Protocol ID: {0}.'.format(protocol))


def send_key(conn: ssl.SSLSocket, protocol, key):
    msg = {
        MessageItems.PROTOCOL: protocol,
        MessageItems.DATA: key
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
            MessageItems.PROTOCOL: protocol,
            MessageItems.USER: token_data['User'],
            MessageItems.TOKEN: token_data['Token']
        }

        conn = self.key_generator.start_connect()
        send_obj(conn, msg)

        msg = receive_obj(conn)

        data = msg[MessageItems.DATA]
        if protocol == Protocols.GET_PKC or protocol == Protocols.GET_PKX:
            ret = paillier.PaillierPublicKey(data)
        elif protocol == Protocols.GET_SKC or protocol == Protocols.GET_SKX:
            pub = paillier.PaillierPublicKey(data[0] * data[1])
            ret = paillier.PaillierPrivateKey(pub, data[0], data[1])
        else:
            ret = None

        msg = {
            MessageItems.PROTOCOL: protocol,
            MessageItems.DATA: 'OK'
        }
        send_obj(conn, msg)
        return ret
