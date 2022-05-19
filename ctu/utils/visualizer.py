"""
Modified from:
  https://github.com/NVlabs/SPADE/blob/master/util/visualizer.py
"""

import os
import time
from io import BytesIO
import ntpath

from ctu.utils import misc, html

class Visualizer:
  def __init__(self, opt):
    self.opt = opt
    self.win_size = opt.display_winsize


  def convert_visuals_to_numpy(self, visuals):
    for key, t in visuals.items():
      tile = self.opt.batch_size > 8
      if key == 'label':
        t = misc.tensor2label(t, self.opt.num_labels + 2, tile=tile)
      else:
        t = misc.tensor2im(t, tile=tile, normalize=True, opt=self.opt)
      visuals[key] = t
    return visuals

  def save_images(self, webpage, visuals, image_path):        
    visuals = self.convert_visuals_to_numpy(visuals)        
      
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims = []
    txts = []
    links = []

    for label, image_numpy in visuals.items():
      image_name = os.path.join(label, '%s.png' % (name))
      save_path = os.path.join(image_dir, image_name)
      misc.save_image(image_numpy, save_path, create_dir=True)

      ims.append(image_name)
      txts.append(label)
      links.append(image_name)
    webpage.add_images(ims, txts, links, width=self.win_size)
  
