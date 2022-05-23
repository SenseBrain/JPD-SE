"""
Data loading for ADE20k.

Modified from:
  https://github.com/NVlabs/SPADE/blob/master/data/ade20k_dataset.py
"""
import sys
import os
from PIL import Image

import numpy as np

from ctu.data.ctu_dataset import CTUDataset
from ctu.data.base_dataset import BaseDataset, get_params, get_transform
from ctu.data.image_folder import make_dataset


class ADE20KDataset(CTUDataset):

  @staticmethod
  def modify_commandline_options(parser, train):
    parser = CTUDataset.modify_commandline_options(parser, train)
    parser.set_defaults(preprocess_mode='fixed')
    parser.set_defaults(load_size=512)
    parser.set_defaults(crop_size=512)
    parser.set_defaults(num_labels=150) 
    parser.set_defaults(contain_dontcare_label=True)
    # parser.set_defaults(no_instance=True)
    return parser


  def get_paths(self, opt):
    root = opt.root_dir
    if opt.mode == 'val':
      root = os.path.join(root, 'validation')
    elif opt.mode == 'test':
      root = os.path.join(root, 'testing')
    else:
      root = os.path.join(root, 'training')
    
    mode = 'val' if opt.mode == 'val' or \
        opt.mode == 'test' else 'train'

    all_images = make_dataset(root, recursive=True)
    image_paths = []
    label_paths = []
    for p in all_images:
      if '_%s_' % mode not in p:
        continue
      if p.endswith('.jpg'):
        image_paths.append(p)
      elif p.endswith('_seg.png'):
        label_paths.append(p)

    instance_paths = label_paths  # the instance seg maps are encoded together w/ the class seg maps in xxx_seg.png

    return label_paths, image_paths, instance_paths


  def postprocess(self, input_dict):
    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    label = input_dict['label']
    label = label - 1
    label[label == -1] = self.opt.num_labels


  def paths_match(self, path1, path2):
    filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
    filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]

    clean_filename1_without_ext = os.path.split(filename1_without_ext)[1]
    clean_filename2_without_ext = os.path.split(filename2_without_ext)[1]

    image_name1 = '_'.join(clean_filename1_without_ext.split('_')[:3])
    image_name2 = '_'.join(clean_filename2_without_ext.split('_')[:3])

    return image_name1 == image_name2


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

    image.save('ade20k_test_img.png')

    params = get_params(self.opt, image.size)
    image = image.convert('RGB')
    transform_image = get_transform(self.opt, params, normalize=True,
        normalize_mean=self.opt.normalize_mean, normalize_std=self.opt.normalize_std)
    image_tensor = transform_image(image)

    if (not self.opt.no_label) or (not self.opt.no_instance):
      # if either is needed, we need to load the images
      seg = Image.open(label_path).convert('RGB')
      seg = np.array(seg)

      #########
      # NOTE channel 0 & 1 have distinct values but they represent the same mask
      # print(label_path, image_path)
      # print(seg[...,0], seg[...,1], seg[...,2])
      # sys.exit()
      #########

      if not self.opt.no_label:
        # R, G channels contain object class masks
        label = Image.fromarray(seg[...,0])
        # label.save('ade20k_test_label0.png')

        # label = Image.fromarray(seg[...,1])
        # label.save('ade20k_test_label1.png')

      if not self.opt.no_instance:
        # B channel contains instance object masks
        instance = Image.fromarray(seg[...,2])
        # instance.save('ade20k_test_label2.png')

    # get label map
    if self.opt.no_label:
      label_tensor = 0
    else:
      # label = Image.open(label_path)

      transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
      label_tensor = transform_label(label) * 255.0

      label_tensor[label_tensor == 255] = self.opt.num_labels  # 'unknown' is opt.num_labels

    # get instance label map 
    if self.opt.no_instance:
      instance_tensor = 0
    else:
      # instance = Image.open(instance_path)

      if self.opt.no_label:
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


if __name__ == '__main__':
  a = ADE20KDataset() 
  a.paths_match(
      'foo/bar/ADE_train_00000975_seg.png',
      'foo/bar/ADE_train_00000975.jpg',
      )
