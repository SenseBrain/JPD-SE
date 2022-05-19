"""
Modified from:
  https://github.com/NVIDIA/pix2pixHD/blob/master/models/models.py
"""

import torch

def create_model(opt):
  if opt.model == 'pix2pixHD':
    from ctu.models.pix2pixHD_model import Pix2PixHDModel, Pix2PixHDInferenceModel
    if opt.is_train:
      model = Pix2PixHDModel()
    else:
      model = Pix2PixHDInferenceModel()
  else:
    # TODO (shiyu) what is this model?
    from ctu.models.ui_model import UIModel
    model = UIModel()
  model.initialize(opt)
  print("model [%s] was created" % (model.__class__.__name__))

  if opt.is_train and len(opt.gpu_ids) and not opt.fp16:
    # TODO
    raise NotImplementedError('we currently do not support multi-gpu training')
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

  return model
