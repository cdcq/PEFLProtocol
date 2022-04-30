import socket
import ssl
import threading


class BaseService:
    def __init__(self, listening: (str, int), cert_path: str, key_path: str,
                 time_out=10, max_receive=1024, max_connection=5):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(listening)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        self.sock = context.wrap_socket(sock, server_side=True)

        self.time_out = time_out
        self.max_receive = max_receive
        self.max_connection = max_connection

    def run(self):
        self.sock.listen(self.max_connection)
        print('Waiting for connection.')
        while True:
            sock, address = self.sock.accept()
            t = threading.Thread(target=self.tcp_handler, args=(sock, address))
            t.start()

    def tcp_handler(self, sock: ssl.SSLSocket, address):
        sock.settimeout(self.time_out)
        sock.close()
