from base_service import BaseService
from connector import Connector
from enums import PROTOCOLS
from key_generator import KeyRequester


class CloudProvider(BaseService, KeyRequester):
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 key_generator: Connector,
                 token_path: str,
                 time_out=10, max_connection=5):
        BaseService.__init__(self, listening, cert_path, key_path,
                             time_out, max_connection)
        KeyRequester.__init__(self, key_generator, token_path)

        self.request_key(PROTOCOLS.GET_SKC)
        self.request_key(PROTOCOLS.GET_PKX)
