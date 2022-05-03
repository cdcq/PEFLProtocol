import json
import ssl
import struct
from phe import paillier


def send_msg(conn: ssl.SSLSocket, msg: bytes):
    conn.sendall(struct.pack('I', len(msg)) + msg)


def send_obj(conn: ssl.SSLSocket, msg):
    send_msg(conn, json.dumps(msg).encode())


def receive_msg(conn: ssl.SSLSocket) -> bytes | None:
    msg_len = struct.unpack('I', conn.recv(4))[0]
    ret = b''
    while len(ret) < msg_len:
        msg = conn.recv(msg_len - len(ret))
        if msg is None:
            return None
        ret = ret + msg

    return ret


def receive_obj(conn: ssl.SSLSocket):
    return json.loads(receive_msg(conn))


def arr_enc(plain: [float], public_key: paillier.PaillierPublicKey, precision=32) \
        -> [paillier.EncryptedNumber]:
    # Ciphertext package.
    ret = [public_key.encrypt(int(i * (2 ** precision))) for i in plain]
    return ret


def arr_enc_len(arr_len: int) -> int:
    return arr_len


def arr_dec(cipher: [paillier.EncryptedNumber], private_key: paillier.PaillierPrivateKey,
            precision=32) -> [float]:
    ret = [private_key.decrypt(i) / (2 ** precision) for i in cipher]
    return ret
