"""This model provide many useful function

The send and receive function can help programmer send a dictionary easily.
The enc and dec function can help programmer encrypt/decrypt an array easily.


Typical usage example:

msg = {
    foo: bar
}
send_obj(conn, msg)

msg = receive_obj(conn)

encrypted = arr_enc(plain, self.public_key)

plain = arr_dec(encrypted, self.private_key)

"""

import json
import ssl
import struct
from math import ceil
from phe import paillier

import torch
from functools import reduce


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


# TODO: add the precision parameter in the use of function.
def arr_enc(plain: [float], public_key: paillier.PaillierPublicKey, package=True, precision=32) \
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


def yield_accumulated_grads(accumulated_grads):
    for i in accumulated_grads:
        yield i


def flatten(generator) -> [float]:
    """
    Get a vector from generator whose element is a matrix of different shapes for different layers of model.
    eg. model.parameters() and yield_accumulated_grads() will create a generator
    for parameters and accumulated_grads
    :param generator:
    :return:
    """

    vector = []
    for para in generator:
        para_list = [i.item() for i in para.flatten()]
        vector.extend(para_list)
    return vector


def de_flatten(vector: [float], model) -> None:
    """
    Use the vector which represents the weights of model to update the values of model.
    :param vector:
    :param model:
    :return:
    """
    with torch.no_grad():
        index = 0
        for para in model.parameters():
            shape = para.shape  # type(shape) is <class 'torch.Size'>, which is a subclass of <class 'tuple'>
            prod_shape = reduce(lambda x, y: x * y, shape)
            para.copy_(torch.tensor(vector[index: (index + prod_shape)]).reshape(shape).requires_grad_())
            index += prod_shape
