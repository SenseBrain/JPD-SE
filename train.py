import os
import time
from collections import OrderedDict

import numpy as np
import torch

import ctu.parsers
import ctu.data
import ctu.trainers
from ctu.utils import html
from ctu.utils.visualizer import Visualizer
from ctu.utils.network_utils import count_params


MAX_VAL_SIZE = 30
parser = ctu.parsers.CTUTrainParser() 
opt = parser.parse()
val_opt = ctu.parsers.trainopt2testopt(opt, mode='val')

print('\ntrain options:\n')
parser.print_options(opt)
print('\nval options:\n')
parser.print_options(val_opt)

# (shiyu) fully deterministic, usually causes slower runtime
if opt.seed:
  import random
  random.seed(opt.seed)
  torch.manual_seed(opt.seed)
  np.random.seed(opt.seed)
  torch.cuda.manual_seed_all(opt.seed)
  torch.backends.cudnn.deterministic=True
  os.environ['PYTHONHASHSEED'] = str(opt.seed)

loader = ctu.data.create_dataloader(opt) 
val_loader = ctu.data.create_dataloader(val_opt)
visualizer = Visualizer(opt)
web_dir = os.path.join(opt.save_dir, 'train_visualizations')
webpage = html.HTML(web_dir, 'visualizations')

trainer = ctu.trainers.get_trainer(opt)(opt, mode='train')

if opt.model == 'pix2pixHD':
  print('# trainable params at initialization: ', count_params(trainer.model) - count_params(trainer.model.netD))
else:
  print('# trainable params at initialization: ', count_params(trainer.model))

if opt.save_dir:
  log_file = os.path.join(opt.save_dir, 'loss_log.txt')

if opt.load_model:
  trainer.load()
  # validate to make sure model-loading has been successful
  with torch.no_grad():
    if opt.save_dir:
      # if load existing model, do not overwrite the original log
      log_mode = 'a' if os.path.isfile(log_file) else 'w'
      print('validating...', file=open(log_file, log_mode))
    print('\nvalidating...\n')
    torch.backends.cudnn.benchmark = False
    distortion_value_total = 0
    start = time.time()
    for i, x_dict in enumerate(val_loader):
      if i == MAX_VAL_SIZE: break
      distortion_value = trainer.get_eval_loss(x_dict)
      distortion_value_total += distortion_value
      end = time.time()
      print('batch {}/{}, distortion ('.format(i+1, len(val_loader)) + opt.distortion_loss_fn + ') {:.4f}, batch processing time (s) {:.4f}'.format(distortion_value, end - start))
      if i > 0:
        start = time.time()
    print('\nvalidation done!\n')
    # distortion_value_avg = distortion_value_total / len(val_loader)
    distortion_value_avg = distortion_value_total / MAX_VAL_SIZE # only using the first MAX_VAL_SIZE imgs from val
    if opt.save_dir:
      print('val set avg distortion (' + opt.distortion_loss_fn + ') {:.4f}'.format(distortion_value_avg), file=open(log_file, 'a'))
    print('\nval set avg distortion (' + opt.distortion_loss_fn + ') {:.4f}\n'.format(distortion_value_avg))
  
if opt.save_dir:
  # if load existing model, do not overwrite the original log
  init_log_mode = 'a' if opt.load_model and os.path.isfile(log_file) else 'w'
torch.backends.cudnn.benchmark = True if not opt.seed else False
for epoch in range(trainer.start_epoch, trainer.start_epoch + opt.num_epochs):
  start = time.time()
  for i, x_dict in enumerate(loader):
    distortion_value = trainer.step(x_dict)
    end = time.time()
    if opt.save_dir and epoch == 0 and i == 0:
      print('epoch {}/{}, batch {}/{}, distortion ('.format(epoch+1, trainer.start_epoch+opt.num_epochs, i+1, len(loader)) + opt.distortion_loss_fn + ') {:.4f}, batch processing time (s) {:.4f}'.format(distortion_value, end - start), file=open(log_file, init_log_mode))
    elif opt.save_dir:
      print('epoch {}/{}, batch {}/{}, distortion ('.format(epoch+1, trainer.start_epoch+opt.num_epochs, i+1, len(loader)) + opt.distortion_loss_fn + ') {:.4f}, batch processing time (s) {:.4f}'.format(distortion_value, end - start), file=open(log_file, 'a'))
    print('epoch {}/{}, batch {}/{}, distortion ('.format(epoch+1, trainer.start_epoch+opt.num_epochs, i+1, len(loader)) + opt.distortion_loss_fn + ') {:.4f}, batch processing time (s) {:.4f}'.format(distortion_value, end - start))
    if i > 0:
      start = time.time()
  
  if not (epoch + 1) % opt.val_interval:
    if opt.save_dir:
      print('validating...', file=open(log_file, 'a'))
    print('\nvalidating...\n')
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
      distortion_value_total = 0
      start = time.time()
      for i, x_dict in enumerate(val_loader):
        if i == MAX_VAL_SIZE: break
        distortion_value = trainer.get_eval_loss(x_dict)
        distortion_value_total += distortion_value 
        end = time.time()
        print('batch {}/{}, distortion ('.format(i+1, len(val_loader)) + opt.distortion_loss_fn + ') {:.4f}, batch processing time (s) {:.4f}'.format(distortion_value, end - start))
        if i > 0:
          start = time.time()
    torch.backends.cudnn.benchmark = True if not opt.seed else False
    print('\nvalidation done!\n')
    
    # distortion_value_avg = distortion_value_total / len(val_loader)
    distortion_value_avg = distortion_value_total / MAX_VAL_SIZE # only using the first MAX_VAL_SIZE imgs from val
    if opt.save_dir:
      print('val set avg distortion (' + opt.distortion_loss_fn + ') {:.4f}'.format(distortion_value_avg), file=open(log_file, 'a'))
    print('\nval set avg distortion (' + opt.distortion_loss_fn + ') {:.4f}\n'.format(distortion_value_avg))
    
    # tf-logging val loss
    if opt.tf_log:
      distortion_value_avg_dict = {'avg_val_distortion': distortion_value_avg}
      trainer.log_loss_values(distortion_value_avg_dict)

    if opt.schedule_lr:
      # perform lr scheduling on the avg val recon loss
      trainer.scheduler_step(distortion_value_avg)

    if opt.always_save or distortion_value_avg < trainer.best_val_loss and opt.save_dir: 
      # save reconstructed image
      print('\nsaving reconstructed val images...\n')
      for i, x_dict in enumerate(val_loader):
        if i == MAX_VAL_SIZE: break # saving all imgs might take a long time
        recon_img = trainer.get_img(x_dict)
        img_path = x_dict['path']
        for j in range(recon_img.shape[0]):
          print('visualizer: processing image {}...'.format(img_path[j]))
          if opt.no_label:
            visuals = OrderedDict([('image', x_dict['image'][j]),
                                 ('reconstructed_image', recon_img[j])])
          else:
            visuals = OrderedDict([('label', x_dict['label'][j]), 
                                 ('image', x_dict['image'][j]),
                                 ('reconstructed_image', recon_img[j])])
          visualizer.save_images(webpage, visuals, img_path[j:j + 1])
        webpage.save()
      print('\nreconstructed val images saved!\n')
      print('saving model...', file=open(log_file, 'a'))
      trainer.save(epoch, distortion_value_avg)
