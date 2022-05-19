"""
Base options.
Modified from:
  https://github.com/NVlabs/SPADE/blob/master/options/base_options.py
"""
import os
import sys
import argparse
import pickle

import torch

import ctu.data as data
import ctu.models as models


def str2bool(s):
  if isinstance(s, bool):
    return s
  if s.lower() in ('true', 't'):
    return True
  elif s.lower() in ('false', 'f'):
    return False
  else:
    raise argparse.ArgumentTypeError('Cannot interpret {} as bool'.format(s))


class CTUParser:
  """CTU parser."""
  def __init__(self):
    self.initialized = False

  def _initialize(self, parser):
    parser.add_argument('--model', type=str, choices=['toderici2017', 'pix2pixHD'])
    parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
    # data I/O
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset', type=str, choices=['ade20k', 'cityscapes', 'coco', 'custom', 'clic'])
    parser.add_argument('--num_workers', type=int, default=4,
        help='Will be fed to the torch DataLoader as the num_workers argument.')
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize,
        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--num_labels', type=int, default=182,
        help='The number of input label classes without unknown class. If you have unknown class as class label, specify --contain_dontcare_label.')
    parser.add_argument('--contain_dontcare_label', action='store_true',
        help='Whether the label map contains dontcare label (dontcare=255)')
    parser.add_argument('--num_out_channels', type=int, default=3,
        help='The number of output image channels. For pix2pixHD, should be equal to input_nc or the discriminator will throw an error')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--no_flip', action='store_true',
        help='If specified, do not flip the images for data argumentation')
    parser.add_argument('--normalize_mean', type=str, default='.5,.5,.5',
        help='A comma-separated 3-number string of means for normalization for images, e.g., ".5,.4,.3". The default is because of the use of the tanh layer as the last layer of the models')
    parser.add_argument('--normalize_std', type=str, default='.5,.5,.5')
    parser.add_argument('--do_not_get_codes', action='store_true',
        help='If specified, do *not* get the binary codes during testing')
    parser.add_argument('--use_gt_semantics', type=str2bool,
                        nargs='?', const=True, default=True,
        help='If specified, use the ground truth semantics instead of the learned ones. If unspecified, need to put the learned semantics in a separate folder named gtFine_learned in the place where gtFine lives. And the learned semantics in gtFine_learned should have the same names as their gt counterparts in gtFine')


    # for displays
    parser.add_argument('--display_winsize', type=int, default=512,
        help='Display window size')

    # preprocessing
    parser.add_argument('--batch_size', type=int, default=1,
        help='Batch size.')
    parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop',
        help='Scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--load_size', type=int, default=1024,
        help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--crop_size', type=int, default=512,
        help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--aspect_ratio', type=float, default=2.0,
        help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--val_preprocess_mode', type=str, default='none',
        help='Scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--val_load_size', type=int, default=1024,
        help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--val_crop_size', type=int, default=512,
        help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--val_aspect_ratio', type=float, default=2.0,
        help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
    parser.add_argument('--test_preprocess_mode', type=str, default='none',
        help='Scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
    parser.add_argument('--test_load_size', type=int, default=1024,
        help='Scale images to this size. The final image will be cropped to --crop_size.')
    parser.add_argument('--test_crop_size', type=int, default=512,
        help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--test_aspect_ratio', type=float, default=2.0,
        help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')

    # add noise at test time
    parser.add_argument('--add_noise', action='store_true', help='if specified, add zero-mean noise to the input images')
    parser.add_argument('--noise_distribution', type=str, default='normal_poisson')
    parser.add_argument('--noise_std', type=float, default=.05, help='std of the noise added to the input image')
    parser.add_argument('--noise_mean', type=float, default=0, help='mean of the noise added to the input image')
    parser.add_argument('--poisson_lambda', type=float, default=.01)

    # model/option saving/loading
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--always_save', action='store_true', help='always save model regardless of the val loss. Useful for training pix2pixHD as a GAN')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--opt_file', type=str, 
        help='The opt file to load from')

    # training
    parser.add_argument('--gpu_ids', type=str, default='0',
        help='A comma separated string of gpu ids: e.g., "0", "0,1". Use "-1" for cpu.')
    parser.add_argument('--num_epochs', type=int, default=100, 
        help='The number of epochs to train')
    parser.add_argument('--val_interval', type=int, default=1, 
        help='The validation interval')
    parser.add_argument('--beta1', type=float, default=0.5,
        help='Momentum term of Adam')
    parser.add_argument('--beta2', type=float, default=0.999,
        help='Momentum term of Adam')
    parser.add_argument('--lr', type=float, default=0.0002,
        help='Initial learning rate for Adam')
    parser.add_argument('--schedule_lr', action='store_true',
        help='Whether to use lr scheduling. Currently only support ReduceLROnPlateau as scheduler')
    parser.add_argument('--lr_decay_factor', type=float, default=.1)
    parser.add_argument('--lr_decay_patience', type=int, default=5)
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--distortion_loss_fn', type=str, default='l1', choices=['l1', 'mse'], help='loss function for calculating the distortion')
     
    self.initialized = True
    return parser
  
  def _gather_options(self):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = self._initialize(parser)

    # get the basic options
    opt, unknown = parser.parse_known_args()

    # modify model-related parser options
    model_name = opt.model
    model_option_setter = models.get_option_setter(model_name)
    parser = model_option_setter(parser, self.is_train)

    # modify dataset-related parser options
    dataset = opt.dataset
    dataset_option_setter = data.get_option_setter(dataset)
    parser = dataset_option_setter(parser, self.is_train)

    opt, unknown = parser.parse_known_args()

    if opt.load_opt:
      print('\nloading opt from {}...\n'.format(opt.opt_file))
      parser = self.update_options_from_file(parser, opt)
      print('\nopt loaded!\n')

    opt = parser.parse_args()
    self.parser = parser
    return opt


  def save_options(self, opt):
    if not os.path.isdir(opt.save_dir):
      os.mkdir(opt.save_dir)
    file_name = os.path.join(opt.save_dir, 'opt')
    with open(file_name + '.txt', 'wt') as f:
      for k, v in sorted(vars(opt).items()):
        comment = ''
        default = self.parser.get_default(k)
        if v != default:
          comment = '\t[default: %s]' % str(default)
        f.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(file_name + '.pkl', 'wb') as f:
      pickle.dump(opt, f)


  def print_options(self, opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
      comment = ''
      default = self.parser.get_default(k)
      if v != default:
        comment = '\t[default: %s]' % str(default)
      message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
  

  def update_options_from_file(self, parser, opt):
    # loading opt only updates the defaults of the existing opts, can be overwritten
    # by explicitly specifying opts
    new_opt = self.load_options(opt)
    for k, v in sorted(vars(opt).items()):
      if hasattr(new_opt, k) and v != getattr(new_opt, k):
        new_val = getattr(new_opt, k)
        parser.set_defaults(**{k: new_val})
    return parser


  def load_options(self, opt):
    new_opt = pickle.load(open(opt.opt_file, 'rb'))
    return new_opt


  def parse(self):
    if self.initialized:
      return self.opt
    
    opt = self._gather_options()
    # (shiyu) self.is_train is set in train_opts.py or test_opts.py
    opt.is_train = self.is_train
    if opt.is_train and opt.save_dir is not None:
      # (shiyu) only saves train opts
      self.save_options(opt)
    
    # only toderici2017 has the semantics_mode param at this point
    # TODO makes more sense to set this in toderici2017 
    if opt.model == 'toderici2017' and opt.semantics_mode == 'none':
      # (shiyu) this makes data loading/preprocessing faster
      opt.no_instance = True
      opt.no_label = True
 
    # parse normalization parameters
    normalize_mean_str = opt.normalize_mean.split(',')
    opt.normalize_mean = [float(mean) for mean in normalize_mean_str]
    normalize_std_str = opt.normalize_std.split(',')
    opt.normalize_std = [float(std) for std in normalize_std_str]

    # parse gpu ids
    gpu_ids_str = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for gpu_id_str in gpu_ids_str:
      gpu_id = int(gpu_id_str)
      if gpu_id >= 0:
        opt.gpu_ids.append(gpu_id)
    if len(opt.gpu_ids) > 0:
      if len(opt.gpu_ids) > 1:
        raise NotImplementedError('Currently we do not support multi-gpu as we tested and found it to be unreasonably slow. Optimizations are being done to make it faster. Meanwhile, if you would like to specify device usage, use the CUDA_VISIBLE_DEVICES flag instead')
      torch.cuda.set_device(opt.gpu_ids[0])
    
    if len(opt.gpu_ids) != 0 and opt.batch_size % len(opt.gpu_ids) != 0:
      raise ValueError('The number of GPUs used must divide batch size, got {} and {}, respectively.'.format(
        len(opt.gpu_ids), opt.batch_size))

    # parse compression quality for the outside codec, if available
    if opt.model == 'pix2pixHD' and opt.use_compressed:
      quality_str = opt.quality.split(',')
      opt.quality = [int(quality) for quality in quality_str]
    
    self.opt = opt
    return self.opt


if __name__ == '__main__':
  pass
