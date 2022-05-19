import os

import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ctu.models.toderici2017_model import Toderici2017Model
from ctu.trainers.base_trainer import BaseTrainer


class Toderici2017Trainer(BaseTrainer):
  # TODO DataParallel
  def __init__(self, opt, mode='train'):
    super(Toderici2017Trainer, self).__init__(opt, mode)
    if len(opt.gpu_ids) > 0:
      self.model = Toderici2017Model(opt).cuda()
    else:
      self.model = Toderici2017Model(opt)

    if self.mode == 'train':
      self.optimizer = self.model.create_optimizers(opt)
      if opt.schedule_lr:
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=opt.lr_decay_factor,
        patience=opt.lr_decay_patience, verbose=True)
    elif self.mode == 'test':
      pass
    else:
      raise ValueError('Invalid trainer mode: {}'.format(self.mode))
  

  def _get_train_loss(self, x_dict):
    states = self.model.get_init_states(opt=self.opt, height=x_dict['image'].size(2), width=x_dict['image'].size(3))
    return self.model(x_dict, states, self.opt, mode='get_train_loss')


  def step(self, x_dict):
    self.train()
    self.optimizer.zero_grad()
    loss = self._get_train_loss(x_dict)
    loss.backward()
    self.optimizer.step()
    self.steps_taken += 1
    if self.opt.tf_log:
      loss_dict = {'train_loss': loss.item()}
      self.log_loss_values(loss_dict)
    return loss.item()

  
  def get_eval_loss(self, x_dict):
    self.eval()
    states = self.model.get_init_states(opt=self.opt, height=x_dict['image'].size(2), width=x_dict['image'].size(3))
    return self.model(x_dict, states, self.opt, mode='get_eval_loss')


  def get_code(self, x_dict):
    self.eval()
    states = self.model.get_init_states(opt=self.opt, height=x_dict['image'].size(2), width=x_dict['image'].size(3))
    return self.model(x_dict, states, self.opt, mode='get_code')


  def get_eval_rate(self, x_dict):
    self.eval()
    states = self.model.get_init_states(opt=self.opt, height=x_dict['image'].size(2), width=x_dict['image'].size(3))
    return self.model(x_dict, states, self.opt, mode='get_eval_rate')


  def get_img(self, x_dict):
    self.eval()
    states = self.model.get_init_states(opt=self.opt, height=x_dict['image'].size(2), width=x_dict['image'].size(3))
    return self.model(x_dict, states, self.opt, mode='get_img')


  def save(self, epoch, val_loss_value):
    self.best_val_loss = val_loss_value
    save_file = os.path.join(self.opt.save_dir, 'checkpoint.pt')
    print('\nsaving checkpoint to {}...\n'.format(save_file))

    states = {
        'epoch': epoch,
        'steps_taken': self.steps_taken,
        'state_dict': self.model.state_dict(),
        'optimizer_state_dict':self.optimizer.state_dict(),
        'best_val_loss': self.best_val_loss
         }
    if self.opt.schedule_lr:
      states['scheduler_state_dict'] = self.scheduler.state_dict()
    torch.save(states, save_file)
    print('\ncheckpoint saved!\n')
  

  def load(self):
    assert os.path.isfile(self.opt.checkpoint_file)
    print('\nloading checkpoint from {}...\n'.format(self.opt.checkpoint_file))
    checkpoint = torch.load(self.opt.checkpoint_file)
   
    try:
      self.model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
      # a hack for loading legacy checkpoints whose weight names start w/ 'codec.'
      new_state_dict = checkpoint['state_dict'].copy()
      for k in checkpoint['state_dict'].keys():
        new_key = '.'.join(k.split('.')[1:])
        new_state_dict[new_key] = new_state_dict.pop(k)
      checkpoint['state_dict'] = new_state_dict
      self.model.load_state_dict(checkpoint['state_dict'])
      
    if self.mode == 'train':
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if self.opt.schedule_lr:
        try:
          self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except KeyError:
          # this is useful in cases where the saved model did not use lr scheduling but one
          # wants to schedule lr for fine-tuning
          print('Did not find scheduler state dict from checkpoint. Not loading it...')
      self.best_val_loss = checkpoint['best_val_loss']
      self.steps_taken = checkpoint['steps_taken']
      # start from the next epoch of the previously saved epoch as we only save 
      # at the end of an epoch
      self.start_epoch = checkpoint['epoch'] + 1
      print('\ncurrent best val loss: {:.4f}\n'.format(self.best_val_loss))
      # all epoch numbers are 0-indexed
      print('\nnow starting from epoch {}...\n'.format(self.start_epoch+1))
    print('\ncheckpoint loaded!\n')
  

  def scheduler_step(self, val_loss_value):
    self.scheduler.step(val_loss_value)


if __name__ == '__main__':
  pass
