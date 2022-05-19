from ctu.parsers.base_parser import *
from ctu.parsers.train_parser import *

def trainopt2testopt(opt, mode='val'):
  """
  args:
    mode: 'val' or 'test'
  """
  if mode not in ['val', 'test']:
    raise ValueError('mode must be either "val" or "test", got {} instead'.format(mode))

  import copy
  new_opt = copy.deepcopy(opt)
  new_opt.mode = mode
  new_opt.is_train = False
  # needs the following because the base dataset creator loads data according to 
  # opt.preprocess_mode, opt.load_size, ... only
  new_opt.preprocess_mode = getattr(opt, mode + '_preprocess_mode')
  new_opt.load_size = getattr(opt, mode + '_load_size')
  new_opt.crop_size = getattr(opt, mode + '_crop_size')
  new_opt.aspect_ratio = getattr(opt, mode + '_aspect_ratio')
  new_opt.batch_size = 1

  # delete the val_xxx and test_xxx attrs as they are no longer needed
  for mode in ['val', 'test']:
    delattr(opt, mode + '_preprocess_mode')
    delattr(opt, mode + '_load_size')
    delattr(opt, mode + '_crop_size')
    delattr(opt, mode + '_aspect_ratio')
    delattr(new_opt, mode + '_preprocess_mode')
    delattr(new_opt, mode + '_load_size')
    delattr(new_opt, mode + '_crop_size')
    delattr(new_opt, mode + '_aspect_ratio')
  return new_opt
