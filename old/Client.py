import ssl
from phe import paillier

from Connector import Connector


class Client(Connector):
    def __init__(self, cloud: (str, int), time_out=10,
                 public_key: paillier.PaillierPublicKey = None,
                 context: ssl.SSLContext = None):
        Connector.__init__(cloud, time_out, public_key, context)

    def encrypt(self, plain_number: int) -> paillier.EncryptedNumber:
        return self.public_key.encrypt(plain_number)
