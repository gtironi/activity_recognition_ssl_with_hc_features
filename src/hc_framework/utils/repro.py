import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # determinismo total custa performance; manter simples para papers
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
