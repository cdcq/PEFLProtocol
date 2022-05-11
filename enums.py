"""This is a constant model

The model contains some constants.

"""


class Protocols:
    GET_PKX = 'GetPKx'
    GET_PKC = 'GetPKc'
    GET_SKX = 'GetSKx'
    GET_SKC = 'GetSKc'

    ROUND_READY = 'RoundReady'
    CLOUD_INIT = 'CloudInit'
    GET_MODEL = 'GetModel'

    SEC_MED = 'SecMed'
    SEC_PER = 'SecPer'
    SEC_AGG = 'SecAgg'
    SEC_EXC = 'SecExc'


class MessageItems:
    PROTOCOL = "Protocol"
    DATA = "Data"
    USER = "User"
    TOKEN = "Token"
    ID = "ID"
