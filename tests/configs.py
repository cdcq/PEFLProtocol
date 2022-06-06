import torch
from random import randint, choice


DATASET = {
    0: "mnist",
    1: "cifar-10",
    2: "CNNDetection",
}
MODEL = {
    # 0: "mlp",
    0: "lenet_mnist",
    1: "lenet_cifar10",
    2: "resnet18_CNNDetect",
}
CALCULATE_MODEL_LENGTH = {
    "mlp": 633226,
    "lenet_mnist": 61706,
    "lenet_cifar10": 62006,
    "resnet19_CNNDetect": 11177538,
}


class Configs:
    DIR_OF_AUTH = "cert"
    KGC_ADDR_PORT = ('127.0.0.1', 8710)
    # CP_ADDR_PORT = ('127.0.0.1', 8711)
    # SP_ADDR_PORT = ('127.0.0.1', 8712)
    CP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))  # 会导致分布式测试失败，cp和edge读出来的ip_port不同
    SP_ADDR_PORT = ('127.0.0.1', randint(1000, 9999))
    TIME_OUT = 3600 * 4

    TASK = 0
    DATASET_NAME = DATASET[TASK]
    MODEL_NAME = MODEL[TASK]
    MODEL_LENGTH = CALCULATE_MODEL_LENGTH[MODEL_NAME]
    TRAINERS_COUNT = 5

    MAX_ROUND = 200
    LEARNING_RATE = 0.01
    CUDA_CHOICES = [i for i in range(torch.cuda.device_count())]
    # DEVICE = torch.device(f"cuda:{choice(CUDA_CHOICES)}") \
    #     if torch.cuda.is_available() else torch.device("cpu")
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    BATCH_SIZE = 32
    POISON_SWAP_LABEL = 1
    SHOW_EDGE_ID = 0

    KEY_SIZE = 2048
    PRECISION = 32
