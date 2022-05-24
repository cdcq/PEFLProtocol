import json
import sys
import os
from ML_utils.model import get_model

sys.path.append(os.path.join(sys.path[0], ".."))
from pefl_protocol.helpers import flatten
from config import Config

if __name__ == "__main__":
    if not os.path.exists("init_weights_vectors"):
        os.mkdir("init_weights_vectors")


    model = get_model(Config.MODEL_NAME)
    init_weights_vector = flatten(model.parameters())


    with open(os.path.join("init_weights_vectors", f"task_{Config.TASK}.txt"), 'w') as write_file:
        json.dump(init_weights_vector, write_file)

    print("Finished writting init model in task:", Config.TASK)
