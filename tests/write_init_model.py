import json
import sys
import os
import yaml
from ML_utils.model import get_model

sys.path.append(os.path.join(sys.path[0], ".."))
from pefl_protocol.helpers import flatten
from configs import Configs


def register_users():
    dir_of_token = os.path.join(Configs.DIR_OF_AUTH, "token")
    if not os.path.exists(dir_of_token):
        os.mkdir(dir_of_token)
    registered_users_tokens = {
        "CP": {"Token": "CP", "Right": 0b10},
        "SP": {"Token": "SP", "Right": 0b00},
        "EDGE": {"Token": "EDGE", "Right": 0b01}
    }
    for i in range(Configs.TRAINERS_COUNT):
        registered_users_tokens[f"EDGE{i}"] = {"Token": f"EDGE{i}", "Right": 0b01}
    with open(os.path.join(Configs.DIR_OF_AUTH, "token", "registered_users.yml"), 'w') as f:
        yaml.safe_dump(registered_users_tokens, f)
    cp_token = {"User": "CP", "Token": "CP"}

    with open(os.path.join(Configs.DIR_OF_AUTH, "token", "cp.yml"), 'w') as f:
        yaml.safe_dump(cp_token, f)

    sp_token = {"User": "SP", "Token": "SP"}
    with open(os.path.join(Configs.DIR_OF_AUTH, "token", "sp.yml"), 'w') as f:
        yaml.safe_dump(sp_token, f)

    for i in range(Configs.TRAINERS_COUNT):
        edge_token = {"User": f"EDGE{i}", "Token": f"EDGE{i}"}
        with open(os.path.join(Configs.DIR_OF_AUTH, "token", f"edge{i}.yml"), 'w') as f:
            yaml.safe_dump(edge_token, f)


if __name__ == "__main__":
    register_users()

    if not os.path.exists("init_weights_vectors"):
        os.mkdir("init_weights_vectors")

    model = get_model(Configs.MODEL_NAME)
    init_weights_vector = flatten(model.parameters())


    with open(os.path.join("init_weights_vectors", f"task_{Configs.TASK}.txt"), 'w') as write_file:
        json.dump(init_weights_vector, write_file)

    print("Finished writting init model in task:", Configs.TASK)
