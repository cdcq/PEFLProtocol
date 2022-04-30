from phe import paillier


def encrypted2dict(number: paillier.EncryptedNumber) -> dict:
    return {
        'c': number.ciphertext(),
        'e': number.exponent
    }


def dict2encrypted(number: dict, pub_key: paillier.PaillierPublicKey) \
        -> paillier.EncryptedNumber:
    return paillier.EncryptedNumber(pub_key, number['c'], number['e'])
