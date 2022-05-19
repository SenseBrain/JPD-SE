import os

import numpy as np

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ctu.models.pix2pixHD_model import Pix2PixHDModel
from ctu.trainers.base_trainer import BaseTrainer

class Pix2PixHDTrainer(BaseTrainer):
  # TODO DataParallel
  def __init__(self, opt, mode='train'):
    super(Pix2PixHDTrainer, self).__init__(opt, mode)
    if len(opt.gpu_ids) > 0:
      self.model = Pix2PixHDModel(opt).cuda()
    else:
      self.model = Pix2PixHDModel(opt)

    if mode == 'train':
      self.optimizer_G, self.optimizer_D = self.model.create_optimizers(opt)
      if self.opt.schedule_lr:
        self.scheduler_G = ReduceLROnPlateau(self.optimizer_G, 'min', factor=opt.lr_decay_factor,
          patience=opt.lr_decay_patience, verbose=True)
        self.scheduler_D = ReduceLROnPlateau(self.optimizer_D, 'min', factor=opt.lr_decay_factor,
          patience=opt.lr_decay_patience, verbose=True)
      self.lambda_distortion_weight = 1.
    elif mode == 'test':
      pass
    else:
      raise ValueError('Invalid trainer mode: {}'.format(mode))
  

  def _get_train_loss(self, x_dict):
    return self.model(x_dict, self.opt, mode='get_train_loss')

  
  def scheduler_step(self, val_loss_value):
    self.scheduler_G.step(val_loss_value)
    self.scheduler_D.step(val_loss_value)


  def step(self, x_dict):
    self.train()
    losses = self._get_train_loss(x_dict)

    loss_dict = dict(zip(self.model.loss_names, losses))
    # calculate final loss scalar
    loss_d_gan = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 if not self.opt.no_d_gan_loss else loss_dict['D_fake'].new_zeros(1, requires_grad=True)
    loss_D = loss_d_gan 
         
    loss_G_gan_feat = loss_dict['G_GAN_Feat'] * self.opt.lambda_feat if not self.opt.no_gan_feat_loss else loss_dict['G_GAN_Feat'].new_zeros(1, requires_grad=True) 
    loss_G_vgg = loss_dict['G_VGG'] * self.opt.lambda_feat if not self.opt.no_vgg_loss else loss_dict['G_VGG'].new_zeros(1, requires_grad=True) 
    loss_G_distortion = loss_dict['G_Distortion'] * self.opt.lambda_distortion * self.lambda_distortion_weight if not self.opt.no_distortion_loss else loss_dict['G_Distortion'].new_zeros(1, requires_grad=True)
    # loss_G = loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict.get('G_Distortion',0)
    loss_G_gan = loss_dict['G_GAN'] if not self.opt.no_g_gan_loss else loss_dict['G_GAN'].new_zeros(1, requires_grad=True)
    loss_G = loss_G_gan + loss_G_vgg + loss_G_gan_feat + loss_G_distortion

    print('g_gan: {:.4f}, g_gan_feat_match: {:.4f}, g_vgg: {:.4f}, g_distortion ('.format(
      loss_dict['G_GAN'].item(), loss_dict['G_GAN_Feat'].item(), loss_dict['G_VGG'].item()) + self.opt.distortion_loss_fn + '): {:.4f}, d_real: {:.4f}, d_fake: {:.4f}'.format(loss_dict['G_Distortion'].item(), 
        loss_dict['D_real'].item(), loss_dict['D_fake'].item() 
  ))

    ############### Backward Pass ####################
    self.optimizer_G.zero_grad()
    if self.opt.fp16:                                
      from apex import amp
      # TODO (shiyu) figure out how this works
      with amp.scale_loss(loss_G, self.optimizer_G) as scaled_loss: scaled_loss.backward()                
    else:
      loss_G.backward()          
    self.optimizer_G.step()

    # update discriminator weights
    self.optimizer_D.zero_grad()
    if self.opt.fp16:                                
      # TODO (shiyu) figure out how this works
      with amp.scale_loss(loss_D, self.optimizer_D) as scaled_loss: scaled_loss.backward()                
    else:
      loss_D.backward()        
    self.optimizer_D.step()        

    self.steps_taken += 1
    if self.opt.anneal_lambda and not (self.steps_taken % self.opt.anneal_interval):
      self.lambda_distortion_weight *= self.opt.anneal_factor
    if self.opt.tf_log:
      self.log_loss_values(loss_dict)
    return loss_dict['G_Distortion'].item()

  
  def get_eval_loss(self, x_dict):
    self.eval()
    with torch.no_grad():
      return self.model(x_dict, self.opt, mode='get_eval_loss').item()


  def _get_code(self, x_dict):
    self.eval()
    with torch.no_grad():
      return self.model(x_dict, self.opt, mode='get_code')


  def get_code(self, x_dict):
    self.eval()
    with torch.no_grad():
      return torch.cat([c for c in self._get_code(x_dict) if c is not None], dim=-1)


  def get_eval_rate(self, x_dict):
    # entropy coding the two codes separately results in a lower bpp
    self.eval()
    with torch.no_grad():
      return self.model(x_dict, self.opt, mode='get_eval_rate')
        

  def get_img(self, x_dict):
    self.eval()
    with torch.no_grad():
      return self.model(x_dict, self.opt, mode='get_img')


  def save(self, epoch, val_loss_value):
    self.best_val_loss = val_loss_value
    save_file = os.path.join(self.opt.save_dir, 'stats_and_optim.pt')
    print('\nsaving checkpoints to {}...\n'.format(self.opt.save_dir))

    states = {
        'epoch': epoch,
        'steps_taken': self.steps_taken,
        'optimizer_G_state_dict':self.optimizer_G.state_dict(),
        'optimizer_D_state_dict':self.optimizer_D.state_dict(),
        'best_val_loss': self.best_val_loss,
        }
    if self.opt.schedule_lr:
      states['scheduler_G_state_dict'] = self.scheduler_G.state_dict()
      states['scheduler_D_state_dict'] = self.scheduler_D.state_dict()
    if self.opt.anneal_lambda:
      states['lambda_distortion_weight'] = self.lambda_distortion_weight
    torch.save(states, save_file)
    self.model.save()
    print('\ncheckpoint saved!\n')


  def load(self):
    print('\nloading checkpoints from {}...\n'.format(self.opt.checkpoints_dir))
    save_file = os.path.join(self.opt.checkpoints_dir, 'stats_and_optim.pt')
    if self.model.use_gpu():
      # TODO support multi-gpu
      stats_and_optim = torch.load(save_file, map_location='cuda:'+str(self.opt.gpu_ids[0]))
    else:
      stats_and_optim = torch.load(save_file, map_location='cpu')
    if self.mode == 'train':
      self.optimizer_G.load_state_dict(stats_and_optim['optimizer_G_state_dict'])
      self.optimizer_D.load_state_dict(stats_and_optim['optimizer_D_state_dict'])
      if self.opt.schedule_lr:
        try:
          self.scheduler_G.load_state_dict(stats_and_optim['scheduler_G_state_dict'])
          self.scheduler_D.load_state_dict(stats_and_optim['scheduler_D_state_dict'])
        except KeyError:
          # this is useful in cases where the saved model did not use lr scheduling but one
          # wants to schedule lr for fine-tuning
          print('Did not find scheduler state dicts from checkpoint. Not loading them...')
      self.best_val_loss = stats_and_optim['best_val_loss']
      self.steps_taken = stats_and_optim['steps_taken']

      if self.opt.anneal_lambda:
        try:
          self.lambda_distortion_weight = stats_and_optim['lambda_distortion_weight']
        except KeyError:
          # this is useful in cases where the saved model did not use lambda annealing but one
          # wants to anneal lambda for fine-tuning
          print('Did not find lambda distortion weight from checkpoint. Not loading it...')
      # start from the next epoch of the previously saved epoch as we only save 
      # at the end of an epoch
      self.start_epoch = stats_and_optim['epoch'] + 1
      print('\ncurrent best val loss: {:.4f}\n'.format(self.best_val_loss))
      # all epoch numbers are 0-indexed
      print('\nnow starting from epoch {}...\n'.format(self.start_epoch+1))
    print('\ncheckpoint loaded!\n')
    # the model will be loaded in pix2pixHDModel
