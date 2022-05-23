"""
Quantization by rounding to the nearest integer. The backward derivative is
approximated by 1. Originally proposed in 'Lossy Image Compression with Compressive Autoencoders'.
"""

import torch

class RoundedIdentity(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return torch.round(input)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.clone()

if __name__ == '__main__':
  x = torch.tensor(1.5, requires_grad=True)
  rid = RoundedIdentity.apply
  y = rid(x)
  y.backward()
  print(x, y, x.grad)
  x = torch.tensor(1.4, requires_grad=True)
  rid = RoundedIdentity.apply
  y = rid(x)
  y.backward()
  print(x, y, x.grad)
  x = torch.tensor(1.6, requires_grad=True)
  rid = RoundedIdentity.apply
  y = rid(x)
  y.backward()
  print(x, y, x.grad)
