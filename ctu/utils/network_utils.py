import torch
import torch.nn as nn

def weights_init(m):
  # fancier init does not seem to help in the current settings
  return
  if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      torch.nn.init.zeros_(m.bias)


def count_params(model):
  # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9?u=michaelshiyu
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
