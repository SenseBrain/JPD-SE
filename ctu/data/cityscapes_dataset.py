"""
Data loading for Cityscapes.

Modified from:
  https://github.com/NVlabs/SPADE/blob/master/data/cityscapes_dataset.py
"""

import os.path
from ctu.data.ctu_dataset import CTUDataset
from ctu.data.image_folder import make_dataset


class CityscapesDataset(CTUDataset):

  @staticmethod
  def modify_commandline_options(parser, train):

    parser = CTUDataset.modify_commandline_options(parser, train)
    parser.set_defaults(preprocess_mode='fixed')
    parser.set_defaults(load_size=512)
    parser.set_defaults(crop_size=512)
    parser.set_defaults(aspect_ratio=2.0)
    parser.set_defaults(num_labels=35)
    
    opt, _ = parser.parse_known_args()

    if hasattr(opt, 'num_upsampling_layers'):
      parser.set_defaults(num_upsampling_layers='more')
      
    return parser

  def get_paths(self, opt):
    root = opt.root_dir
    mode = opt.mode

    if opt.use_gt_semantics:
      print('using gt semantics...')
      label_dir = os.path.join(root, 'gtFine', mode)
    else:
      print('using learned semantics...')
      label_dir = os.path.join(root, 'gtFine_learned', mode)
    label_paths_all = make_dataset(label_dir, recursive=True)
    label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]

    image_dir = os.path.join(root, 'leftImg8bit', mode)
    image_paths = make_dataset(image_dir, recursive=True)

    if not opt.no_instance:
      instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
    else:
      instance_paths = []

    return label_paths, image_paths, instance_paths

  def paths_match(self, path1, path2):
    name1 = os.path.basename(path1)
    name2 = os.path.basename(path2)
    # compare the first 3 components, [city]_[id1]_[id2]
    return '_'.join(name1.split('_')[:3]) == \
           '_'.join(name2.split('_')[:3])
