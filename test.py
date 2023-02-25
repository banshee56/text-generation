from torch import Tensor
import torch

m = torch.empty((2, 5))
m = m.uniform_(-5.0, 5.0)
print(m)