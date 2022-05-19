"""
Modified from:
  https://github.com/NVIDIA/pix2pixHD/blob/master/models/base_model.py
"""

import os
import sys

import torch

class BaseModel(torch.nn.Module):

  def __init__(self, opt):
    super(BaseModel, self).__init__()
    self.opt = opt
    self.gpu_ids = opt.gpu_ids
    self.is_train = opt.is_train
    self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor


  def set_input(self, input):
    self.input = input


  def forward(self):
    pass


  # used in test time, no backprop
  def test(self):
    pass


  def get_image_paths(self):
    pass


  def optimize_parameters(self):
    pass


  def get_current_visuals(self):
    return self.input


  def get_current_errors(self):
    return {}


  def save(self, label):
    pass


  def save_network(self, network, network_label, opt):
    save_filename = 'net_%s.pth' % (network_label)
    save_path = os.path.join(opt.save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
      network.cuda()


  def load_network(self, network, network_label, opt):    
    load_filename = 'net_%s.pth' % (network_label)
    load_path = os.path.join(opt.checkpoints_dir, load_filename)    
    if not os.path.isfile(load_path):
      print('%s does not exist' % load_path)
      if network_label == 'G':
        raise('generator must exist')
    else:
      #network.load_state_dict(torch.load(load_path))
      try:
        network.load_state_dict(torch.load(load_path))
      except:   
        pretrained_dict = torch.load(load_path)          
        model_dict = network.state_dict()
        try:
          pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}            
          network.load_state_dict(pretrained_dict)
          print('pretrained network %s has excessive layers. Only loading layers that are used' % network_label)
        except:
          print('pretrained network %s has fewer layers. The following are not initialized:' % network_label)
          for k, v in pretrained_dict.items():            
            if v.size() == model_dict[k].size():
              model_dict[k] = v

          if sys.version_info >= (3,0):
            not_initialized = set()
          else:
            from sets import Set
            not_initialized = Set()            

          for k, v in model_dict.items():
            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
              not_initialized.add(k.split('.')[0])
          
          print(sorted(not_initialized))
          network.load_state_dict(model_dict)          

  def update_learning_rate(self, ):
    pass
