import torch
from torch.autograd import Function

class Sign(Function):
    """
    It's based on VRIC-RNN paper
    https://arxiv.org/abs/1511.06085
    """

    def __init__(self):
        super(Sign, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        # Only training mode quantization noise is applied
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None