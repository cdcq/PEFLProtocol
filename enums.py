from enum import Enum, auto


class PROTOCOLS(Enum):
    GET_PKX = auto()
    GET_PKC = auto()
    GET_SKX = auto()
    GET_SKC = auto()

    ROUND_READY = auto()
