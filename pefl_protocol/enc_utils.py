from phe import paillier
from math import ceil
from multiprocessing import Pool

from pefl_protocol.configs import Configs


def gen_ciphertext(enc_number: [paillier.EncryptedNumber]) -> [int]:
    return [i.ciphertext() for i in enc_number]


class Encryptor:
    def __init__(self, public_key: paillier.PaillierPublicKey,
                 private_key: paillier.PaillierPrivateKey = None,
                 precision=32, value_range_bits=16,
                 key_length=2048, if_package=True):
        self.pub = public_key
        self.prv = private_key
        self.precision = precision
        self.value_bits = value_range_bits
        self.if_package = if_package
        self.key_length = key_length

    def arr_enc_len(self, arr_len: int) -> int:
        if self.if_package:
            padding = self.precision
            number_length = self.precision + self.value_bits + self.precision
            numbers_per_package = self.key_length // (number_length + padding)
            return ceil(arr_len / numbers_per_package)
        else:
            return arr_len

    def enc_number(self, number: int) -> paillier.EncryptedNumber:
        return self.pub.encrypt(number)

    def arr_enc(self, plain: [float]) -> [paillier.EncryptedNumber]:
        if self.if_package is True:
            padding = self.precision
            number_length = self.precision + self.value_bits + self.precision
            numbers_per_package = self.key_length // (number_length + padding)

            base = 1
            power = 2 ** (number_length + padding)
            number_range = 2 ** number_length
            prc_power = 2 ** self.precision

            packages = []  # Attention, ret will append a 0 at the start of loop.
            for i in range(len(plain)):
                if i % numbers_per_package == 0:
                    packages.append(0)
                    base = 1
                number = int(plain[i] * prc_power)
                if number < 0:
                    # This is complement way to express minus.
                    # If number < 0, then number is not 0, so here is no need
                    # to mod number_range.
                    number = number_range + number
                packages[-1] = packages[-1] + number * base
                base = base * power

            # ret = [self.pub.encrypt(i) for i in packages]
            with Pool(Configs.PROCESS_COUNT) as pool:
                ret = pool.map(self.enc_number, packages)
        else:
            prc_power = 2 ** self.precision
            ret = [self.pub.encrypt(int(i * prc_power)) for i in plain]

        return ret

    def dec_number(self, cipher: paillier.EncryptedNumber) -> int:
        return self.prv.decrypt(cipher)

    def arr_dec(self, cipher: [paillier.EncryptedNumber], arr_len) -> [float]:
        if self.if_package is True:
            padding = self.precision
            number_length = self.precision + self.value_bits + self.precision
            numbers_per_package = self.key_length // (number_length + padding)

            power = 2 ** (number_length + padding)
            sign_flag = 2 ** (number_length - 1)
            number_range = 2 ** number_length
            prc_power = 2 ** self.precision

            # plain = [self.prv.decrypt(i) for i in cipher]
            with Pool(Configs.PROCESS_COUNT) as pool:
                plain = pool.map(self.dec_number, cipher)

            numbers = []
            for i in plain:
                for j in range(numbers_per_package):
                    number = i % power
                    i = i // power
                    # Attention, number must be modeled by range even if
                    # it is positive.
                    number = number % number_range
                    # Attention!! Here couldn't use ">=" to judge if minus.
                    # if number >= sign_flag:
                    if number & sign_flag:
                        number = -(number_range - number)
                    numbers.append(number)

            numbers = numbers[:arr_len]
            ret = [i / prc_power for i in numbers]
        else:
            prc_power = 2 ** self.precision
            ret = [self.prv.decrypt(i) / prc_power for i in cipher]

        return ret

    def gen_enc_number(self, ciphertext: [int]) -> [paillier.EncryptedNumber]:
        return [paillier.EncryptedNumber(self.pub, i) for i in ciphertext]
