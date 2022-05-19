"""
Implementing the network components for the codec proposed in
'Full resolution image compression with recurrent neural networks' by Toderici et al.

Modified from:
  https://github.com/chaoyuaw/pytorch-vcii
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ConvRNNCellBase(nn.Module):
  def __repr__(self):
    s = (
        '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
        ', stride={stride}')
    if self.padding != (0, ) * len(self.padding):
      s += ', padding={padding}'
    if self.dilation != (1, ) * len(self.dilation):
      s += ', dilation={dilation}'
    s += ', hidden_kernel_size={hidden_kernel_size}'
    s += ')'
    return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):
  def __init__(self,
               input_channels,
               hidden_channels,
               kernel_size=3,
               stride=1,
               padding=0,
               dilation=1,
               hidden_kernel_size=1,
               bias=True):
    super(ConvLSTMCell, self).__init__()
    self.input_channels = input_channels
    self.hidden_channels = hidden_channels

    self.kernel_size = _pair(kernel_size)
    self.stride = _pair(stride)
    self.padding = _pair(padding)
    self.dilation = _pair(dilation)

    self.hidden_kernel_size = _pair(hidden_kernel_size)

    hidden_padding = _pair(hidden_kernel_size // 2)

    gate_channels = 4 * self.hidden_channels
    self.conv_ih = nn.Conv2d(
        in_channels=self.input_channels,
        out_channels=gate_channels,
        kernel_size=self.kernel_size,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
        bias=bias)

    self.conv_hh = nn.Conv2d(
        in_channels=self.hidden_channels,
        out_channels=gate_channels,
        kernel_size=hidden_kernel_size,
        stride=1,
        padding=hidden_padding,
        dilation=1,
        bias=bias)

    self.reset_parameters()


  def reset_parameters(self):
    self.conv_ih.reset_parameters()
    self.conv_hh.reset_parameters()


  def forward(self, x, hidden):
    hx, cx = hidden
    gates = self.conv_ih(x) + self.conv_hh(hx)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


class EncoderCell(nn.Module):
  def __init__(self, in_channels):
    super(EncoderCell, self).__init__()

    self.conv = nn.Conv2d(
        in_channels, 
        256, # was 64 
        kernel_size=3, stride=2, padding=1, bias=False)

    self.rnn1 = ConvLSTMCell(
        256, # was 64
        256,
        kernel_size=3,
        stride=2,
        padding=1,
        hidden_kernel_size=1,
        bias=False)

    self.rnn2 = ConvLSTMCell(
        256,
        512,
        kernel_size=3,
        stride=2,
        padding=1,
        hidden_kernel_size=1,
        bias=False)

    self.rnn3 = ConvLSTMCell(
        512,
        512,
        kernel_size=3,
        stride=2,
        padding=1,
        hidden_kernel_size=1,
        bias=False)


  def forward(self, x, hidden1, hidden2, hidden3):
    x = self.conv(x)
    hidden1 = self.rnn1(x, hidden1)
    x = hidden1[0]
    hidden2 = self.rnn2(x, hidden2)
    x = hidden2[0]
    hidden3 = self.rnn3(x, hidden3)
    x = hidden3[0]

    return x, hidden1, hidden2, hidden3


class DecoderCell(nn.Module):
  def __init__(self, bin_channels, out_channels):

    super(DecoderCell, self).__init__()

    self.conv1 = nn.Conv2d(
        bin_channels, 512, kernel_size=1, stride=1, padding=0, bias=False)

    self.rnn1 = ConvLSTMCell(
        512,
        512,
        kernel_size=3,
        stride=1,
        padding=1,
        hidden_kernel_size=1,
        bias=False)

    self.rnn2 = ConvLSTMCell(
        128, #out1=256
        512,
        kernel_size=3,
        stride=1,
        padding=1,
        hidden_kernel_size=1,
        bias=False)

    self.rnn3 = ConvLSTMCell(
        128, #out2=128
        256,
        kernel_size=3,
        stride=1,
        padding=1,
        hidden_kernel_size=3,
        bias=False)

    self.rnn4 = ConvLSTMCell(
        64, #out3=64
        128,
        kernel_size=3,
        stride=1,
        padding=1,
        hidden_kernel_size=3,
        bias=False)

    self.conv2 = nn.Conv2d(
        32,
        out_channels, 
        kernel_size=1, stride=1, padding=0, bias=False)


  def forward(self, x, hidden1, hidden2, hidden3, hidden4):
    x = self.conv1(x)
    hidden1 = self.rnn1(x, hidden1)

    x = hidden1[0]
    x = F.pixel_shuffle(x, 2)

    hidden2 = self.rnn2(x, hidden2)

    x = hidden2[0]
    x = F.pixel_shuffle(x, 2)

    hidden3 = self.rnn3(x, hidden3)

    x = hidden3[0]
    x = F.pixel_shuffle(x, 2)

    hidden4 = self.rnn4(x, hidden4)

    x = hidden4[0]
    x = F.pixel_shuffle(x, 2)

    x = torch.tanh(self.conv2(x)) / 2

    return x, hidden1, hidden2, hidden3, hidden4


def init_lstm(batch_size, height, width):

    encoder_h_1 = (
        torch.zeros(batch_size, 256, height // 4, width // 4, requires_grad=False),
        torch.zeros(batch_size, 256, height // 4, width // 4, requires_grad=False))
    encoder_h_2 = (
        torch.zeros(batch_size, 512, height // 8, width // 8, requires_grad=False),
        torch.zeros(batch_size, 512, height // 8, width // 8, requires_grad=False))
    encoder_h_3 = (
        torch.zeros(batch_size, 512, height // 16, width // 16, requires_grad=False),
        torch.zeros(batch_size, 512, height // 16, width // 16, requires_grad=False))

    decoder_h_1 = (
        torch.zeros(batch_size, 512, height // 16, width // 16, requires_grad=False),
        torch.zeros(batch_size, 512, height // 16, width // 16, requires_grad=False))
    decoder_h_2 = (
        torch.zeros(batch_size, 512, height // 8, width // 8, requires_grad=False),
        torch.zeros(batch_size, 512, height // 8, width // 8, requires_grad=False))
    decoder_h_3 = (
        torch.zeros(batch_size, 256, height // 4, width // 4, requires_grad=False),
        torch.zeros(batch_size, 256, height // 4, width // 4, requires_grad=False))
    decoder_h_4 = (
        torch.zeros(batch_size, 256 if False else 128, height // 2, width // 2, requires_grad=False),
        torch.zeros(batch_size, 256 if False else 128, height // 2, width // 2, requires_grad=False))

    return (encoder_h_1, encoder_h_2, encoder_h_3, 
            decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)


if __name__ == '__main__':
  t = EncoderCell()
  it = DecoderCell(256)
  print(repr(t))
  print(repr(it))

  init_lstm(100, 64, 64)
