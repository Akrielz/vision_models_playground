from torch import nn

from vision_models_playground.components.activations import QuickGELU


class QuickGEGLU(nn.Module):
    def __init__(self):
        super(QuickGEGLU, self).__init__()
        self.quick_gelu = QuickGELU()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * self.quick_gelu(gates)
