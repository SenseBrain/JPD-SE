"""
Implementation of the codec described in 'Soft-to-Hard Vector Quantization 
for End-to-End Learning Compressible Representations'.

A schematic overview of this codec is given below.

                                          
  transform                  encode                             
x --------> continuous repr. -----> discrete, compressible* code
                                                         |
                                                         |entropy
                                                         |encode
                                                         |(lossless)
                                                         | 
                                                         V
x <------ continuous repr. <----- discrete code <------ bitstream
  inverse                  decode               entropy
  transform                                     decode
                                                (lossless)

*meaning that the underlying distribution has low entropy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ctu.quantizers.s2h_vq import S2HVQ, S2HVQV2
from ctu.utils.network_utils import weights_init


class GDN(nn.Module):
  def __init__(self, n_channel):
    super(GDN, self).__init__()
    self.fc = nn.Linear(n_channel, n_channel, bias=True)
  
  def forward(self, x, inverse):
    """
    args:
      x: tensor (shape: (n, c, h, w))

    returns:
      x_normalized: tensor (shape: (n, c, h, w))
    """
    for _ in self.parameters():
      _.data.clamp_(0.)
    x_channel_last = x.permute(0, 2, 3, 1).pow(2)
    coef = self.fc(x_channel_last).pow(.5).permute(0, 3, 1, 2)
    if inverse:
      return x * coef
    else:
      return x / coef

class ResBlock(nn.Module):
  def __init__(self):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels=128, 
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        )
    self.conv2 = nn.Conv2d(
        in_channels=128, 
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        )
    self.bn1 = nn.BatchNorm2d(128)
    self.bn2 = nn.BatchNorm2d(128)
    
  def forward(self, x):
    y = self.conv1(x)
    y = F.leaky_relu(y)
    y = self.conv2(x)
    return x + y

class S2HTransform(nn.Module):
  """
  For input image with shape (128, 128, 3), this transform maps
  it to an image with shape (16, 16, channels[2]). This follows
  what has been described in the paper.
  """
  
  def __init__(self, channel=128):
    super(S2HTransform, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=128,
        kernel_size=5,
        stride=2,
        padding=2,
        dilation=1,
        )
    self.conv2 = nn.Conv2d(
        in_channels=128,
        out_channels=128,
        kernel_size=5,
        stride=2,
        padding=2,
        dilation=1,
        )
    self.conv3 = nn.Conv2d(
        in_channels=128, 
        out_channels=channel,
        kernel_size=5,
        stride=2,
        padding=2,
        dilation=1,
        )
    # conv1/2/3 downsamples height or width by 2 if height or width 
    # is even. If odd, outputs size (height or width + 1)/2

    self.resblock1 = ResBlock()
    self.resblock2 = ResBlock()
    self.resblock3 = ResBlock()

  def forward(self, x, flatten=True):
    x = F.leaky_relu(self.conv1(x))
    x = F.leaky_relu(self.conv2(x)) 
    x = self.resblock1(x) 
    x = self.resblock2(x)
    x = self.resblock3(x)
    x = torch.tanh(self.conv3(x))

    return x

class S2HInvTransform(nn.Module):
  """
  Upsamples the input image by 8x. For an input image with shape
  (16, 16, channels[0]), this operation maps it to an image with
  shape (128, 128, 3).
  """
  
  def __init__(self, channel=128):
    super(S2HInvTransform, self).__init__()
    self.tconv1 = nn.ConvTranspose2d(
        in_channels=channel,
        out_channels=128,
        kernel_size=3, 
        stride=2, 
        padding=1, 
        output_padding=1, 
        )

    self.tconv2 = nn.ConvTranspose2d(
        in_channels=128,
        out_channels=128,
        kernel_size=3, 
        stride=2, 
        padding=1, 
        output_padding=1, 
        )
    
    self.tconv3 = nn.ConvTranspose2d(
        in_channels=128,
        out_channels=3,
        kernel_size=3, 
        stride=2, 
        padding=1, 
        output_padding=1, 
        )

    self.resblock1 = ResBlock()
    self.resblock2 = ResBlock()
    self.resblock3 = ResBlock()
  
  def forward(self, x):
    x = self.tconv1(x) 
    x = self.resblock1(x) 
    x = self.resblock2(x)
    x = self.resblock3(x)
    x = F.leaky_relu(self.tconv2(x)) 
    x = torch.tanh(self.tconv3(x))
    x = .5 * (x + 1)
    x = x.clamp(0, 1)
    return x

class S2HCodec(nn.Module):

  def __init__(self, code_book, channel=128, sigma=1.):
    super(S2HCodec, self).__init__()
    # initialize encoder/decoder module
    # center_size must divide x.size(1)
    self._channel = channel
    self.quantizer = S2HVQ(code_book=code_book, sigma=sigma)
    
    self.transform = S2HTransform(channel)
    self.inv_transform = S2HInvTransform(channel)

    self.apply(weights_init)

  @property
  def channel(self):
    return self._channel
  
  @property
  def code_book(self):
    return self.quantizer.code_book

  @property
  def sigma(self):
    return self.quantizer.sigma

  @sigma.setter
  def sigma(self, new_sigma):
    self.quantizer.sigma = new_sigma

  def t(self, x, flatten=True):
    """
    t(ransform).
    
    args:
      x: tensor (shape: (n, c, h, w))
      flatten (optional): bool
        If true, return the flattened actiations of x.
    """
    x = self.transform(x)
    if flatten:
      return x.view(x.size(0), -1)
    else:
      return x

  def te(self, x, train=True, raw=False):
    """
    t + e(ncode).

    args:
      raw: (optional) bool
        If true, encode each x[i] into a sequence of vectors, each containing
        the score over all centers in the code book. If false, encode each 
        x[i] into a sequence of scalars, each containing the index of the center
        with the highest score in the code book.
    """
    z = self.t(x, flatten=True)
    if raw:
      code = self.quantizer.encode(
          x=z, 
          code_len=z.size(1)//self.quantizer.code_book.size(1),
          train=train,
          raw=True
          )
    else:
      code = self.quantizer.encode(
          x=z, 
          code_len=z.size(1)//self.quantizer.code_book.size(1),
          train=train,
          raw=False
          )
    return code
 
  def di(self, code_raw, target_shape):
    """
    d + i.

    args:
      code_raw: tensor (shape: (n, code_len, n_center))
        The output from te(..., raw=True).
      target_shape: list or tuple of ints
        The shape into which the decoded vector will be reshaped before being
        fed into the inverse transform. This should be the shape of 
        t(x, flatten=False).
    """
    decoded = self.quantizer.decode(code_raw)

    decoded_reshaped = decoded.view(target_shape)
    return self.inv_transform(decoded_reshaped)
  
  def ted(self, x, train=True):
    """
    t + e + d(ecode).
    """
    return self.quantizer.decode(self.te(x, train=train, raw=True))

  def tedi(self, x, train=True):
    """
    t + e + d + i(nverse transform).
    """
    z = self.t(x, flatten=False)
    code_raw = self.quantizer.encode(
          x=z, 
          code_len=z.size(1)//self.quantizer.code_book.size(1),
          train=train,
          raw=True
          )
    target_shape = z.size()

    decoded = self.quantizer.decode(code_raw=code_raw)

    decoded_reshaped = decoded.view(target_shape)
    return self.inv_transform(decoded_reshaped)

class S2HCodecV2(S2HCodec):
  # TODO tests
  def __init__(self, **kwargs):
    super(S2HCodecV2, self).__init__(**kwargs)
    self.quantizer = S2HVQV2(**kwargs)
    self.apply(weights_init)

if __name__ == '__main__':
  pass 
