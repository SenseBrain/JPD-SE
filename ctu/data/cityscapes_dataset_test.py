import sys

import numpy as np
import skimage.io as io

import ctu.data as data
import ctu.opts

opt = CTUTrainParser().parse() 
dataloader = data.create_dataloader(opt)

for x in dataloader:
  label, img, ins, path = x['label'], x['image'], x['instance'], x['path']
  print(path)
  print('label size', label.size())
  print('image size', img.size())
  print('instance label size', ins.size())
  io.imshow(np.round(((img[0].permute(1, 2, 0).numpy()) * .5  + .5) * 255).astype(np.uint8))
  io.show()
  sys.exit()
