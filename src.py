import torch
from torch import nn, optim
from typing import List, Dict
import random
import numpy as np
from collections import Counter
import time
import os

print(torch.__version__)
print(torch.cuda.is_available())

seed = 12345
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

