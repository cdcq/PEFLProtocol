"""This is a constant model

The model contains some constants.

"""


from enum import Enum, auto


class Protocols(Enum):
    GET_PKX = auto()
    GET_PKC = auto()
    GET_SKX = auto()
    GET_SKC = auto()

    ROUND_READY = auto()
    CLOUD_INIT = auto()
    GET_MODEL = auto()

    SEC_MED = auto()
    SEC_PER = auto()
    SEC_AGG = auto()
    SEC_EXC = auto()


class MessageItems(Enum):
    PROTOCOL = "Protocol"
    DATA = "Data"
    USER = "User"
    TOKEN = "Token"
    ID = "ID"
