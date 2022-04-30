import socket
import ssl

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 2000)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_default_certs()
ssock = context.wrap_socket(sock, server_hostname='127.0.0.1')

