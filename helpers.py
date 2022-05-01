import json
import ssl
import struct


def send_msg(sock: ssl.SSLSocket, msg: bytes):
    sock.sendall(struct.pack('I', len(msg)) + msg)


def send_obj(sock: ssl.SSLSocket, msg):
    send_msg(sock, json.dumps(msg).encode())


def receive_msg(sock: ssl.SSLSocket) -> bytes | None:
    msg_len = struct.unpack('I', sock.recv(4))[0]
    ret = b''
    while len(ret) < msg_len:
        msg = sock.recv(msg_len - len(ret))
        if msg is None:
            return None
        ret = ret + msg

    return ret


def receive_obj(sock: ssl.SSLSocket):
    return json.loads(receive_msg(sock))
