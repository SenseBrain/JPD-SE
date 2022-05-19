import re

import torch
from torch.utils.data import DataLoader

def get_mean_n_std(dataset):
  """
  Get the sample mean of a set of tensor over each channel.
  Adapted from 
  https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6?u=michaelshiyu 
  
  args:
    dataset: torch.utils.data.DataLoader
      Dataset with channel number c.

  returns: 
    m: tensor (shape: (c, ))
    std: tensor (shape: (c, ))
  """
  data_loader = DataLoader(
      dataset,
      batch_size=10,
      num_workers=1,
      shuffle=False
      )
  mean = 0.
  std = 0.
  n_example = 0
  for x in data_loader:
    x = x.view(x.size(0), x.size(1), -1)
    mean += x.mean(2).sum(0)
    std += x.std(2).sum(0)
    n_example += x.size(0)

  mean /= n_example
  std /= n_example

  return mean, std


def get_mean_n_std_ctu(dataloader):
  """
  Get the sample mean of a set of tensor over each channel.
  Adapted from 
  https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6?u=michaelshiyu 
  
  args:
    dataloader: torch.utils.data.DataLoader 
      A ctu dataloader.

  """
  img_mean = 0.
  img_std = 0.
  label_mean = 0.
  label_std = 0.
  instance_mean = 0.
  instance_std = 0.
  n_example = 0

  for i, x_dict in enumerate(dataloader):
    print('{}/{}'.format(i, len(dataloader)))
    img = x_dict['image']

    # sanity check for cityscapes: uncommenting the following two lines
    # should result in a zero-mean, unit-var set
    print(img.size())
    # img = img - torch.tensor([0.2869, 0.3252, 0.2839]).view(1, 3, 1, 1)
    # img = img / torch.tensor([0.1761, 0.1810, 0.1777]).view(1, 3, 1, 1)

    img = img.view(img.size(0), img.size(1), -1)
    img_mean += img.mean(2).sum(0)
    img_std += img.std(2).sum(0)
    n_example += img.size(0)
    
    label = x_dict['label'] / 255.
    
    # sanity check for cityscapes: uncommenting the following two lines
    # should result in a zero-mean, unit-var set
    # label = label - torch.tensor(0.047334)
    # label = label / torch.tensor(0.027811)

    label = label.view(label.size(0), -1)
    label_mean += label.mean(1).sum(0)
    label_std += label.std(1).sum(0)

    instance = x_dict['instance'] / 255.
    # sanity check for cityscapes: uncommenting the following two lines
    # should result in a zero-mean, unit-var set
    # instance = instance - torch.tensor()
    # instance = instance / torch.tensor()
    instance = instance.view(instance.size(0), -1)
    print(instance.max())
    instance_mean += instance.mean(1).sum(0)
    instance_std += instance.std(1).sum(0)

    
  label_mean /= n_example
  label_std /= n_example
  img_mean /= n_example
  img_std /= n_example
  instance_mean /= n_example
  instance_std /= n_example

  print('label mean {}, label std {}, instance mean {}, instance std {}, img mean {}, img std {}'.format(label_mean, label_std, instance_mean, instance_std, img_mean, img_std))


def atoi(text):
  """
  From:
    https://github.com/NVlabs/SPADE/blob/master/util/util.py
  """
  return int(text) if text.isdigit() else text


def natural_keys(text):
  """
  From:
    https://github.com/NVlabs/SPADE/blob/master/util/util.py

  Comment from the original implementation:
  alist.sort(key=natural_keys) sorts in human order
  http://nedbatchelder.com/blog/200712/human_sorting.html
  (See Toothy's implementation in the comments)
  """ 
  return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
  """
  From:
    https://github.com/NVlabs/SPADE/blob/master/util/util.py
  """
  items.sort(key=natural_keys)

if __name__ == '__main__':
  pass
