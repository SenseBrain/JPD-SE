"""
Generic data loading.

Modified from:
  https://github.com/NVlabs/SPADE/blob/master/data/pix2pix_dataset.py
"""

import sys
import os
import time
from PIL import Image

import numpy as np
import skimage.io as io

import torch

from ctu.data.base_dataset import BaseDataset, get_params, get_transform
import ctu.utils.preprocessing as preprocessing


class CTUDataset(BaseDataset):
  """
  The standard abstract dataset that all concrete datasets should inherit from.
  """
  @staticmethod
  def modify_commandline_options(parser, train):
    parser.add_argument('--no_pairing_check', action='store_true',
                        help='If specified, skip sanity check of correct label-image file pairing')
    return parser

  def initialize(self, opt):
    self.opt = opt
    label_paths, image_paths, instance_paths = self.get_paths(opt)

    preprocessing.natural_sort(label_paths)
    preprocessing.natural_sort(image_paths)
    if not opt.no_instance:
      preprocessing.natural_sort(instance_paths)

    label_paths = label_paths[:opt.max_dataset_size]
    image_paths = image_paths[:opt.max_dataset_size]
    instance_paths = instance_paths[:opt.max_dataset_size]

    if not opt.no_pairing_check:
      for path1, path2 in zip(label_paths, image_paths):
        if not self.paths_match(path1, path2):
          raise ValueError(
              "The label-image pair {}, {} do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/ctu_dataset.py to see what is going on, and use --no_pairing_check to bypass this.".format(path1, path2))

    self.label_paths = label_paths
    self.image_paths = image_paths
    self.instance_paths = instance_paths

    size = len(self.label_paths)
    self.dataset_size = size

  def get_paths(self, opt):
    """
    returns:
      label_paths: list
      image_paths: list
      instance_paths: list
    """
    raise NotImplementedError(
        "A subclass of CTUDataset must override self.get_paths(self, opt)")

  def paths_match(self, path1, path2):
    filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
    filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
    return filename1_without_ext == filename2_without_ext

  def __getitem__(self, index):
    image_path = self.image_paths[index]
    # sanity checks
    if not self.opt.no_label:
      label_path = self.label_paths[index]
      if not self.paths_match(label_path, image_path):
        raise ValueError(
        "The label_path {} and image_path {} don't match.".format(
          label_path, image_path))
    if not self.opt.no_instance:
      instance_path = self.instance_paths[index]
      if not self.paths_match(instance_path, image_path):
        raise ValueError(
        "The instance_path {} and image_path {} don't match.".format(
          instance_path, image_path))

    # get img 
    image = Image.open(image_path)
    params = get_params(self.opt, image.size)
    image = image.convert('RGB')
    transform_image = get_transform(self.opt, params, normalize=True,
        normalize_mean=self.opt.normalize_mean, normalize_std=self.opt.normalize_std)
    image_tensor = transform_image(image)

    # get label map
    if self.opt.no_label:
      label_tensor = 0
    else:
      label = Image.open(label_path)

      transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
      label_tensor = transform_label(label) * 255.0
      label_tensor[label_tensor == 255] = self.opt.num_labels  # 'unknown' is opt.num_labels

      # (shiyu) conversion to one-hot tensors is much faster on GPU, so perform this 
      # preprocessing in model's preprocess method instead
      
    # get instance label map 
    if self.opt.no_instance:
      instance_tensor = 0
    else:
      instance = Image.open(instance_path)

      if self.opt.no_label:
        # (shiyu) instance label map and label map use the same transform
        transform_ins = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
      else:
        transform_ins = transform_label

      if instance.mode == 'L':
        instance_tensor = transform_ins(instance) * 255
        instance_tensor = instance_tensor.long()
      else:
        instance_tensor = transform_ins(instance)

    input_dict = {'label': label_tensor,
                  'instance': instance_tensor,
                  'image': image_tensor,
                  'path': image_path,
                  }
    
    # Give subclasses a chance to modify the final output
    self.postprocess(input_dict)

    return input_dict

  def postprocess(self, input_dict):
    return input_dict

  def __len__(self):
    return self.dataset_size

if __name__ == '__main__':
  path1 = 'foo/imgs/berlin_1_img.png'
  path2 = 'foo/labels/berlin_1_label.png'
  filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
  filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
  print(filename1_without_ext, filename2_without_ext)
