import torch
from random import randint

DATASET = {
    0: "mnist",
    1: "cifar-10",
    2: "CNNDetection",
}
MODEL = {
    0: "mlp",
    2: "resnet18_CNNDetect"
}
CALCULATE_MODEL_LENGTH = {
    0: 633226,
    2: 11177538
}


class Config:
    DIR_OF_AUTH = "cert"
    KGC_ADDR_PORT = ('127.0.0.1', 8710)
    CP_ADDR_PORT = ('127.0.0.1', 8711)
    SP_ADDR_PORT = ('127.0.0.1', 8712)
    # KGC_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    # CP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    # SP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    TIME_OUT = 3600

    TASK = 2
    DATASET_NAME = DATASET[TASK]
    MODEL_NAME = MODEL[TASK]
    MODEL_LENGTH = CALCULATE_MODEL_LENGTH[TASK]
    TRAINERS_COUNT = 1
    MAX_ROUND = 100
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 32

    POISON_SWAP_LABEL = 1
    PRECISION = 32


