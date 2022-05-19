"""
Modified from:
  https://github.com/NVlabs/SPADE/blob/master/data/__init__.py
"""

import importlib

import torch.utils.data
from ctu.data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
  # Given the option --dataset [datasetname],
  # the file "datasets/datasetname_dataset.py"
  # will be imported. 
  dataset_filename = "ctu.data." + dataset_name + "_dataset"
  datasetlib = importlib.import_module(dataset_filename)

  # In the file, the class called DatasetNameDataset() will
  # be instantiated. It has to be a subclass of BaseDataset,
  # and it is case-insensitive.
  dataset = None
  target_dataset_name = dataset_name.replace('_', '') + 'dataset'
  for name, cls in datasetlib.__dict__.items():
    if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
      dataset = cls
            
  if dataset is None:
    raise ValueError("In {}.py, there should be a subclass of BaseDataset "
                     "with class name that matches {} in lowercase.".format(
                       dataset_filename, target_dataset_name))

  return dataset


def get_option_setter(dataset_name):    
  dataset_class = find_dataset_using_name(dataset_name)
  return dataset_class.modify_commandline_options


def create_dataloader(opt):
  dataset = find_dataset_using_name(opt.dataset)
  instance = dataset()
  instance.initialize(opt)
  print("dataset [{}] of size {} was created".format(
    type(instance).__name__, len(instance)))

  dataloader = torch.utils.data.DataLoader(
    instance,
    batch_size=opt.batch_size,
    shuffle=opt.is_train,
    num_workers=int(opt.num_workers),
    drop_last=opt.is_train
  )
  return dataloader
