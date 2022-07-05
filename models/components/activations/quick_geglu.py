from torch import nn

from models.components import QuickGELU


class QuickGEGLU(nn.Module):
    def __init__(self):
        super(QuickGEGLU, self).__init__()
        self.gelu = QuickGELU()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * self.gelu(gates)
