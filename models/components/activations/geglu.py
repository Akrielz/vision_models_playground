from torch import nn
from torch.nn import functional as F


class GEGLU(nn.Module):
    def __init__(self):
        super(GEGLU, self).__init__()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
