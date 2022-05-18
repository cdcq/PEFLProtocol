import torch

def poisoned_local_update(model, dataloader,
                 lr=0.01, momentum=0.0, local_eps=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    pass
