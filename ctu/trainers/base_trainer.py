import os

import torch

class BaseTrainer(torch.nn.Module):
  def __init__(self, opt, mode='train'):
    super(BaseTrainer, self).__init__()
    self.opt = opt
    if mode == 'train':
      self.steps_taken = 0 # the total number of train steps taken
      self.start_epoch = 0
      self.best_val_loss = 1e12
      if opt.tf_log:
        import tensorflow as tf
        self.tf = tf
        tf_log_dir = os.path.join(opt.save_dir, 'tf_log')
        self.writer = tf.summary.FileWriter(tf_log_dir)
    elif mode == 'test':
      pass
    else:
      raise ValueError('Invalid trainer mode: {}'.format(mode))
    self.mode = mode
    

  def load(self):
    pass


  def save(self, epoch, val_loss_value):
    pass


  def get_img(self, x_dict):
    """
    Returns the reconstructed image without grad history.
    """
    self.eval()
    pass


  def _get_train_loss(self, x_dict):
    """
    Returns the loss term used for training.
    """
    self.train()
    pass


  def step(self, x_dict):
    """
    Take a training step and return a loss value.

    Note that there may be several loss terms to minimize, but this method
    should always return only one scalar which will be printed in train.py as a summary for this step. This scalar should correspond to
    the validation set value used for model-saving and lr-scheduling.

    One can optionally print out the values of other loss terms inside step.
    """
    self.train()
    pass


  def get_eval_loss(self, x_dict):
    """
    Returns a loss value with no grad history.

    Note that there may be several loss terms involved, but this method
    should always return only one scalar which will be used for 
    model-saving and lr-scheduling.

    One can optionally print out the values of other loss terms inside step.
    """
    self.eval()
    pass


  def scheduler_step(self, val_loss_value):
    pass

  def log_loss_values(self, loss_dict):
    for k, v in loss_dict.items():
      try:
        # some trainers pass torch tensors whereas others pass python numbers
        summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=k, simple_value=v.item())])
      except AttributeError:
        summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=k, simple_value=v)])
      self.writer.add_summary(summary, self.steps_taken)
