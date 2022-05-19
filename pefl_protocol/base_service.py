"""This is the base class for the function of server part in protocol.

The class have a SSL context object to communicate. You do not have to provide the context object but
just need to provide the socket that the server will listen. The class will create a new context object
automatically. You also need to provide the certificate file and the private key file for SSL.

"""

import socket
import ssl
import threading


class BaseService:
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 time_out=60, max_connection=5):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(listening)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        self.sock = context.wrap_socket(sock, server_side=True)

        self.time_out = time_out
        self.max_connection = max_connection

    def run(self):
        self.sock.listen(self.max_connection)
        print('Waiting for connection.')
        while True:
            conn, address = self.sock.accept()
            t = threading.Thread(target=self.tcp_handler, args=(conn, address))
            t.start()

    def tcp_handler(self, conn: ssl.SSLSocket, address):
        """
        This function will always be rewrote at sub classes.
        """

        conn.settimeout(self.time_out)
        conn.close()
