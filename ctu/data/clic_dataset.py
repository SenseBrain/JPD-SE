"""
Data loading for CLIC.

Modified from:
  https://github.com/NVlabs/SPADE/blob/master/data/cityscapes_dataset.py
"""

import os.path
from ctu.data.ctu_dataset import CTUDataset
from ctu.data.image_folder import make_dataset


class ClicDataset(CTUDataset):

  @staticmethod
  def modify_commandline_options(parser, train):

    parser = CTUDataset.modify_commandline_options(parser, train)
    parser.set_defaults(preprocess_mode='none')
    parser.set_defaults(num_labels=54)

    opt, _ = parser.parse_known_args()

    if hasattr(opt, 'num_upsampling_layers'):
      parser.set_defaults(num_upsampling_layers='more')
      
    return parser

  def get_paths(self, opt):
    root = opt.root_dir
    mode = opt.mode

    label_dir = os.path.join(root, mode, 'sem')
    label_paths_all = make_dataset(label_dir, recursive=True)
    label_paths = [p for p in label_paths_all if p.endswith('_sem_map.png')]

    image_dir = os.path.join(root, mode, 'img')
    image_paths = make_dataset(image_dir, recursive=True)

    if not opt.no_instance:
      instance_paths = [p for p in label_paths_all if p.endswith('_ins_map.png')]
    else:
      instance_paths = []

    return label_paths, image_paths, instance_paths

  def paths_match(self, path1, path2):
    # path2 is the path to the image, path1 the semantics
    name1 = os.path.basename(path1)
    name2 = os.path.basename(path2)
    return name1.startswith(os.path.splitext(name2)[0])
