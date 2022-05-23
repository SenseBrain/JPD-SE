"""
Implementation of the binary quantizer in 'Variable Rate Image Compression with 
Recurrent Neural Networks' (https://arxiv.org/abs/1511.06085).

Adapted from https://github.com/chaoyuaw/pytorch-vcii. 
"""
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

class SoftSignFunction(Function):

  def __init__(self):
    super(SoftSignFunction, self).__init__()

  @staticmethod
  def forward(ctx, input):
    prob = input.new(input.size()).uniform_()
    x = input.clone()
    x[(1 - input) / 2 <= prob] = 1
    x[(1 - input) / 2 > prob] = -1
    return x

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output


class DifferentiableSign(nn.Module):
  
  def __init__(self, ):
    super(DifferentiableSign, self).__init__()

  def forward(self, x):
    # Apply quantization noise while only training
    if self.training:
      return SoftSignFunction.apply(x)
    else:
      return x.sign()


class Binarizer(nn.Module):

  def __init__(self, in_channels, out_channels, groups=1):
    super(Binarizer, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, groups=groups)
    self.differentiable_sign = DifferentiableSign()

  def forward(self, x):
    x = self.conv(x)
    x = torch.tanh(x)
    x = self.differentiable_sign(x)
    
    """
    # assuming examples within a batch are all of the same size
    num_bits = torch.tensor(x.view(x.size(0), -1).size(-1))
    p = torch.mean((x.view(x.size(0), -1) + 1) / 2., dim=-1)
    entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    num_bits_post_entropy_coding = entropy * num_bits.to(torch.float).to(entropy.device)
    print('total # bits: {}'.format([num_bits] * x.size(0)))
    print('total # bits after entropy coding (estimated): {}'.format(num_bits_post_entropy_coding.data))
    """
    return x


if __name__ == '__main__':
  binarizer = DifferentiableSign()
  # binarizer.eval()
  binarizer.train()
  x = 2 * (np.random.rand(10, 3) - .5)
  x = torch.tensor(x, requires_grad=True)
  binarized_x = binarizer(x)
  print(x)
  print(binarized_x)
  binarized_x.sum().backward()
  print(x.grad)
