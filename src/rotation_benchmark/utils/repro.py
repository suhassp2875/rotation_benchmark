import os
import random
import numpy as np
import torch

def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not support deterministic mode depending on version/device
            pass
