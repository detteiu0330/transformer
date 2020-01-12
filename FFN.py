import torch
from torch import nn as N


class FeedForward(N.Module):
    def __init__(self, output_size):
        super(FeedForward, self).__init__()
        self.FFN1 = N.Linear(output_size, output_size * 4)
        self.FFN2 = N.Linear(output_size * 4, output_size)

    def forward(self, inputs):
        inputs = self.FFN1(inputs)
        inputs = torch.relu(inputs)
        outputs = self.FFN2(inputs)
        return outputs
