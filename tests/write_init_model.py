import json
import sys
import os
from ML_utils.model import get_model

sys.path.append(os.path.join(sys.path[0], ".."))
from pefl_protocol.helpers import flatten
from configs import Configs

if __name__ == "__main__":
    model = get_model(Configs.MODEL_NAME)
    init_weights_vector = flatten(model.parameters())
    with open(os.path.join("init_weights_vectors", f"task_{Configs.TASK}.txt"), 'w') as write_file:
        json.dump(init_weights_vector, write_file)

    print("Finished init model in task:", Configs.task)
