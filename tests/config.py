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
    KGC_ADDR_PORT = ('127.0.0.1', 8700)
    CP_ADDR_PORT = ('127.0.0.1', 8701)
    SP_ADDR_PORT = ('127.0.0.1', 8702)
    # KGC_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    # CP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    # SP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    TIME_OUT = 300

    TASK = 0
    DATASET_NAME = DATASET[TASK]
    MODEL_NAME = MODEL[TASK]
    MODEL_LENGTH = CALCULATE_MODEL_LENGTH[TASK]
    TRAINERS_COUNT = 3
    MAX_ROUND = 1000
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 32

    POISON_SWAP_LABEL = 1

