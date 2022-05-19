"""
Data loading for custom dataset.  
"""

import os.path
from ctu.data.pix2pix_dataset import Pix2pixDataset
from ctu.data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):

  @staticmethod
  def modify_commandline_options(parser, train):

    parser = Pix2pixDataset.modify_commandline_options(parser, train)
    parser.set_defaults(preprocess_mode='fixed')
    parser.set_defaults(load_size=512)
    parser.set_defaults(crop_size=512)
    parser.set_defaults(aspect_ratio=2.0)
    parser.set_defaults(normalize_mean='0., 0., 0.')
    parser.set_defaults(normalize_std='1., 1., 1.')
    
    opt, _ = parser.parse_known_args()

    return parser

  def get_paths(self, opt):
    root = opt.root_dir
    mode = opt.mode

    # label_dir = os.path.join(root, 'gtFine', mode)
    # label_paths_all = make_dataset(label_dir, recursive=True)
    # label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]

    image_dir = os.path.join(root, mode)
    image_paths = make_dataset(image_dir, recursive=True)

    # if not opt.no_instance:
    #   instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
    # else:
    #   instance_paths = []

    label_paths = image_paths
    instance_paths = image_paths
    return label_paths, image_paths, instance_paths

  def paths_match(self, path1, path2):
    name1 = os.path.basename(path1)
    name2 = os.path.basename(path2)
    # compare the first 3 components, [city]_[id1]_[id2]
    return '_'.join(name1.split('_')[:3]) == \
           '_'.join(name2.split('_')[:3])
