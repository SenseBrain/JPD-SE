"""
PyTorch implementation of the en/decoding module from 'Soft-to-Hard Vector 
Quantization for End-to-End Learning Compressible Representations'.
"""
import sys
from collections import OrderedDict 

import numpy as np

import torch
import torch.nn as nn

class S2HVQ(nn.Module):
  """
  Soft-to-hard vector quantization for en/decoding as described in 'Soft-
  to-Hard Vector Quantization for End-to-End Learning Compressible 
  Representations'.

  Reference table of our notations vs. the original notations in the paper:
    ours        paper

    n           sample size (not used in the paper)
    d           d
    code_len    m
    n_center    L
    center_size d//m (assumes m divides d)
    x           z
    x_mtrx      Z
    x_disc      phi(Z)(:=[phi(z_bar^(1)), ..., phi(z_bar^(m))]) or phi_hat(Z)
    code_book   script_C
    x_recons_mtrx Z_hat
    x_recons    z_hat
    sigma       sigma
  """
  def __init__(self, code_book, sigma=10., **kwargs):
    """
    args:
      code_book: tensor (shape: (n_center, center_size))
        The set of vector centers to perform vector quantization with.
        As described in the paper, this code book is learnable and will 
        hence be saved as this module's parameter.
      
      sigma: float
        Hyperparameter controlling the degree of 'hardness' of the soft
        quantization. Must be positive. Only effective in training. 
        Refer to the paper for more details.
    """
    super(S2HVQ, self).__init__()
    assert sigma > 0, "sigma must be greater than 0, got {}".format(sigma)

    self._center_size = code_book.size(1)
    self._code_book = nn.Parameter(code_book)
    self._sigma = sigma

  @property
  def center_size(self):
    return self._center_size

  @property
  def code_book(self):
    return self._code_book
  
  @property
  def sigma(self):
    return self._sigma

  @sigma.setter
  def sigma(self, new_sigma):
    assert new_sigma > 0, "sigma must be greater than 0, got {}".format(new_sigma)
    self._sigma = new_sigma
  
  def _get_score_mtrx(self, x_mtrx):
    """
    Computes distance matrix between each vector component of x_mtrx and
    all centers. More concretely, we have
      score_mtrx[i][j][k] = L2_dist(x_mtrx[i][j], code_book[k]).

    args:
      x_mtrx: tensor (shape: (n, code_len, center_size))
        The matrix representation of input x to be encoded.

    returns:
      score_mtrx: tensor (shape: (n, code_len, n_center))
    """
    score_mtrx = (x_mtrx.unsqueeze(dim=2) - self.code_book).pow_(2)
    # score_mtrx.size() -> n * code_len * n_center * center_size 
    score_mtrx = score_mtrx.sum(dim=-1)
    # score_mtrx.size() -> n * code_len * n_center 
    return score_mtrx

  def _soft_quantize(self, x_mtrx):
    """
    Soft quantization as described in the paper.

    args:
      x_mtrx: tensor
        The matrix representation of input x to be encoded.

    returns:
      x_disc: tensor (dtype: float; shape: (n, code_len, n_center))
        The continuous code for input x.

      About the notation: for soft quantization, the returned tensors are
      in fact of type float. We use the name x_disc(rete) for consistency 
      with the _hard_quantize method.
    """
    score_mtrx = self._get_score_mtrx(x_mtrx)
    x_disc = torch.softmax(score_mtrx.mul_(-self.sigma), dim=-1)
    return x_disc

  def _hard_quantize(self, x_mtrx):
    """
    Hard quantization as described in the paper.

    args:
      x_mtrx: tensor
        The matrix representation of input x to be encoded.

    returns:
      x_disc: tensor (dtype: int; shape: (n, code_len, n_center))
        The discrete code for input x.
    """
    score_mtrx = self._get_score_mtrx(x_mtrx)
    # TODO: tie-breaking
    _, min_index = torch.min(score_mtrx, dim=-1, keepdim=True)
    x_disc = x_mtrx.new_zeros(
        [x_mtrx.size(0), x_mtrx.size(1), self.code_book.size(0)])

    # convert to one-hot representation to be consistent with _soft_quantize
    x_disc.scatter_(dim=2, index=min_index, value=1)
    return x_disc
  
  def _vec2mtrx(self, x, code_len):
    """
    The reshaping procedure converting each input vector to its matrix 
    representation. In the paper, this is the transformation from z to Z.

    args:
      x: tensor (shape: (n, d))
        The vector to be reshaped. 
        
        NOTE: We know that x is in fact a matrix, by vector, we 
        mean that each input, i.e., x[i], for each i, is a vector :)
      code_len: int
        Desired column number of the resulting matrix representation.
        This will be the code length of the final discrete code.
        Must divide d.

    returns:
      x_mtrx: tensor (shape: (n, code_len, d // code_len))
        The matrix representation of x.
    """ 
    return x.view(-1, code_len, x.size(1) // code_len)
  
  def _mtrx2vec(self, x_mtrx):
    """
    Converting each input from its matrix representation to its original
    vector representation. In the paper, this is the transformation from
    Z to z.

    args:
      x_mtrx: tensor (shape: (n, code_len, d // code_len))

    returns:
      x: tensor (shape: (n, d))
    """
    return x_mtrx.view(-1, x_mtrx.size(1) * x_mtrx.size(2))
  
  def _decode_mtrx(self, code_raw):
    """
    Decode the raw code of each input, returns the matrix representation of 
    the decoded input.

    args:
      code_raw: tensor (shape: (n, code_len, n_center)) 
        The compressible representation to be decoded. Each code_raw[i][j] contains
        the scores of all centers in the code book.

    returns:
      decoded_mtrx: tensor (shape: (n, code_len, center_size))
        The decoded symbol with matrix representation.
    """
    _, max_index = torch.max(code_raw, dim=-1)
    decoded_mtrx = self.code_book[max_index[:][:]]
    
    # this is the soft decoding used in the paper, but the actual method used in this 
    # implementation, i.e., always perform hard decoding, was found by us to work better
    
    # decoded_mtrx = torch.matmul(code_raw, self.code_book)
    return decoded_mtrx

  def decode(self, code_raw):
    """
    Decode the code of each input, returns the original vector representation of 
    the decoded input.
    
    args:
      code_raw: tensor (shape: (n, code_len, n_center)) 
        The compressible representation to be decoded. Each code_raw[i][j] contains
        the scores of all centers in the code book.
        This could be the output from encode(..., raw=True).

    returns:
      x: tensor (shape: (n, d))
        The decoded vector.
    """
    decoded_mtrx = self._decode_mtrx(code_raw=code_raw)
    decoded = self._mtrx2vec(x_mtrx=decoded_mtrx)
    return decoded

  def _encode_sclr(self, x, code_len, train=True):
    """
    Encode each x[i] into a sequence of code_len integers.

    args:
      x: tensor (shape: (n, d)) 
        The vector to be encoded.
      code_len: int
        The desired length of the code for x. Must divide x.size(1).       
      train (optional): bool
        If true, use (differentiable) soft quantization.

    returns:
      code: tensor (shape: (n, code_len)) 
        The discrete integer-sequence code of x.
    """
    # TODO test this function
    code_raw = self._encode_vctr(x=x, code_len=code_len, train=train)
    _, code = torch.max(code_raw, dim=-1)
    return code

  def _encode_vctr(self, x, code_len, train=True):
    """
    Encode each x[i] into a sequence of code_len vectors, each with dimension
    n_center.

    args:
      x: tensor (shape: (n, d)) 
        The vector to be encoded.
      code_len: int
        The desired length of the code for x.
        Must divide x.size(1).       
      train (optional): bool
        If true, use (differentiable) soft quantization.

    returns:
      code_raw: tensor (shape: (n, code_len, n_center)) 
        The raw code for x, i.e., x_disc[i][j] contains the scores of all
        centers in the code book.
    """
    x_mtrx = self._vec2mtrx(x=x, code_len=code_len)
    
    if train:
      code_raw = self._soft_quantize(x_mtrx=x_mtrx)
    else:
      code_raw = self._hard_quantize(x_mtrx=x_mtrx)
    return code_raw
  
  def encode(self, x, code_len, train=True, raw=True):
    """
    Encode each input vector into a compressible code stream.

    args:
      x: tensor (shape: (n, d)) 
        The vector to be encoded.
      code_len: int
        The desired length of the code for x.
        Must divide x.size(1).       
      train (optional): bool
        If true, use (differentiable) soft quantization.
      raw (optional): bool
        If true, outputs the full score over all centers in the code book.
        If false, outputs the index of the center with the highest score.

    returns:
      code_raw if raw else code: tensor (shape: if raw, (n, code_len, n_center); else, (n, code_len)) 
        The code for x, i.e., x_disc[i][j] contains the scores of all
        centers in the code book if raw is set to true, it contains the index
        of the center with the highest score if raw is false.
    """
    # TODO deprecate code_len: code_len is something the alg. can infer from the shape
    # of x and that of the code_book. In that case, only one sanity
    # check needs to be performed upon receiving x, which is to 
    # check that x.size(1) can be divided by code_book.size(1).
    if x.size(1) < code_len:
      raise ValueError(
          "x.size(1) must be greater than or equal to code_len, got " +\
              "{} and {}, respectively".format(x.size(1), code_len))
    if x.size(1) % code_len != 0:
      raise ValueError(
          "code_len must divide x.size(1), got {} and {}, respectively".format(
            code_len, x.size(1)))
    if x.size(1) // code_len != self.code_book.size(1):
      raise ValueError("Illegal code_len. x.size(1) // code_len must equal " +\
          "code_book.size(0), got {}, {}, and {}, respectively.".format(
            x.size(1), code_len, self.code_book.size(1)))
    
    if raw:
      return self._encode_vctr(x=x, code_len=code_len, train=train)
    else:
      return self._encode_sclr(x=x, code_len=code_len, train=train)

  @staticmethod
  def get_pmf(scores):
    """
    Compute the histogram over the centers of the code book as an estimate to 
    the probability mass function (pmf). The histogram is computed over the 
    batch of inputs.
    
    args:
      scores: tensor (shape: (n, code_len, n_center))
        Scores over the centers. May use the output from encode(..., raw=True).

    returns:
      pmf: tensor (shape: (n_center, ))
        The estimated probability mass over the centers of the code book.
    """
    pmf = scores.sum(dim=(0, 1)) / (scores.size(0) * scores.size(1))
    return pmf
  
  @staticmethod
  def get_cross_entropy(pmf1, pmf2):
    """
    Compute the batch cross-entropy estimate of the pmf over the 
    centers in the code book.
    Using the paper's notations, this method computes H(pmf1, pmf2).
    
    args:
      pmf1(2): tensor (shape: (n_center, ))
        The pmf estimates. May use the output of get_pmf.

    returns:
      cross_entropy: float
        The estimated cross_entropy.
    """
    assert torch.all(pmf1 >= 0), "pmf1 have < 0 or nan element(s)"
    assert torch.all(pmf2 >= 0), "pmf2 have < 0 or nan element(s)"
    assert torch.allclose(
        pmf1.sum(), 
        torch.tensor(1., device=pmf1.device)), "pmf1 is not normalized"
    assert torch.allclose(
        pmf2.sum(), 
        torch.tensor(1., device=pmf2.device)), "pmf2 is not normalized"
    pmf2_positive = pmf2[pmf2 > 0]
    pmf1_positive = pmf1[pmf2 > 0]
    
    cross_entropy_raw = -pmf1_positive * torch.log2(pmf2_positive)
    return cross_entropy_raw.sum()

class S2HVQV2(S2HVQ):
  # TODO tests
  def __init__(self, **kwargs):
    super(S2HVQV2, self).__init__(**kwargs)
    self.net = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(self.code_book.size(1), 64)),
        ('gate1', nn.Tanh()),
        ('fc2', nn.Linear(64, 128)),
        ('gate2', nn.Tanh()),
        ('fc3', nn.Linear(128, self.code_book.size(0))),
        ]))
  
  def _get_score_mtrx(self, x_mtrx):
    return self.net(x_mtrx)

if __name__ == '__main__':
  # the following test script is more informative than the corresponding tests
  # in the test suite
  """  
  code_book = torch.randn(10, 5)
  s2h = S2HVQ(code_book, sigma=1)
  x = torch.randn(100, 15)
  # s2h.encode(x=x, code_len=5)
  # s2h.encode(x=x, code_len=20)
  # s2h.encode(x=x, code_len=7)
  """
  
