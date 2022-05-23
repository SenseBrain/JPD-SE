"""
Modified from:
  https://github.com/NVlabs/SPADE/blob/master/test.py
"""

import os
import time
from collections import OrderedDict

import numpy as np
import torch

import pytorch_msssim

import ctu.parsers
import ctu.data
import ctu.trainers
from ctu.utils import html
from ctu.utils.visualizer import Visualizer
from ctu.utils.misc import tensor2im


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = ctu.parsers.CTUTrainParser()
opt = ctu.parsers.trainopt2testopt(parser.parse(), mode='test')
print('\ntest options:\n')
parser.print_options(opt)

loader = ctu.data.create_dataloader(opt)
trainer = ctu.trainers.get_trainer(opt)(opt, mode='test') 

# load_model
trainer.load()

visualizer = Visualizer(opt)
# create a webpage that summarizes the all results
web_dir = os.path.join(opt.save_dir, 'test_visualizations')
webpage = html.HTML(web_dir, 'visualizations')

print('\ntesting...\n')
distortion_value_L1_total = 0.
distortion_value_MSE_total = 0.
distortion_value_MSSSIM_total = 0.
if not opt.do_not_get_codes:
  code_shannon_bpp_total = 0.
  code_actual_bpp_total = 0.

loss_fn_l1 = torch.nn.L1Loss()
loss_fn_mse = torch.nn.MSELoss()
loss_fn_msssim = pytorch_msssim.MSSSIM()

if not opt.do_not_get_codes:
  if not os.path.exists(os.path.join(opt.save_dir, 'codes')):
    os.mkdir(os.path.join(opt.save_dir, 'codes'))

with torch.no_grad():
  start = time.time()
  for i, x_dict in enumerate(loader):

    if opt.add_noise:
      mx, mn = torch.max(x_dict['image']), torch.min(x_dict['image'])
      if 'poisson' in opt.noise_distribution:
        # need to denormalize the image to >= 0
        raise NotImplementedError()
        print('adding poisson noise...')
        x_dict['image'] += opt.poisson_lambda * torch.normal(opt.noise_mean * torch.ones_like(x_dict['image']), torch.sqrt(x_dict['image']))
      if 'normal' in opt.noise_distribution:
        print('adding normal noise...')
        x_dict['image'] += torch.zeros_like(x_dict['image']).normal_(opt.noise_mean, opt.noise_std)
      if 'uniform' in opt.noise_distribution:
        raise NotImplementedError()
        # x_dict['image'] += x_dict['image'].new_zeros(x_dict['image'].size()).uniform_(-.5 * 12 ** .5 * opt.noise_std, .5 * 12 ** .5 * opt.noise_std) 
      x_dict['image'] = torch.clamp(x_dict['image'], mn, mx)

    # get eval loss value

    if not opt.do_not_get_codes:
      code_shannon_bpp, code_actual_bpp = trainer.get_eval_rate(x_dict)
      code_shannon_bpp_total += code_shannon_bpp
      code_actual_bpp_total += code_actual_bpp

    # save reconstructed image
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


      if not opt.do_not_get_codes:
        codes = trainer.get_code(x_dict)
        base_name = os.path.splitext(os.path.split(img_path[j])[1])[0]
        code_name = os.path.join(opt.save_dir, 'codes', base_name+'_code')
        # np.savetxt(code_name, codes[j].cpu().numpy().astype(np.uint8), delimiter='', fmt='%d')
        with open(code_name, 'wb') as f:
          # print(codes.size())
          try:
            f.write(codes.cpu().numpy().astype(np.uint8).tobytes())
          except AttributeError:
            for code in codes:
              if code is not None:
                f.write(code.cpu().numpy().astype(np.uint8).tobytes())
      
    webpage.save()
    
    # this denormalizes the images and quantize them so that the loss values would be accurate
    recon_img = torch.tensor(tensor2im(recon_img, opt).transpose(0, 3, 1, 2)).to(torch.float)
    real_img = torch.tensor(tensor2im(x_dict['image'], opt).transpose(0, 3, 1, 2)).to(torch.float)
    if trainer.model.use_gpu():
      recon_img = recon_img.cuda()
      real_img = real_img.cuda()
    distortion_value_L1 = loss_fn_l1(recon_img, real_img).item()
    distortion_value_L1_total += distortion_value_L1
    distortion_value_MSE = loss_fn_mse(recon_img, real_img).item()
    distortion_value_MSE_total += distortion_value_MSE
    distortion_value_MSSSIM = loss_fn_msssim(recon_img, real_img).item()
    distortion_value_MSSSIM_total += distortion_value_MSSSIM

    end = time.time()
    if not opt.do_not_get_codes:
      print('batch {}/{}, recon loss (L1/MSE/MS-SSIM) {:.4f}/{:.4f}/{:.4f}, pre-/(estimated) post-entropy coding bpp {:.4f}/{:.4f}, batch processing time (s) {:.4f}'.format(
      i+1, len(loader), distortion_value_L1, distortion_value_MSE, distortion_value_MSSSIM, code_actual_bpp, code_shannon_bpp, end - start))
    else:
      print('batch {}/{}, recon loss (L1/MSE/MS-SSIM) {:.4f}/{:.4f}/{:.4f}, batch processing time (s) {:.4f}'.format(
      i+1, len(loader), distortion_value_L1, distortion_value_MSE, distortion_value_MSSSIM, end - start))
    if i > 0:
      start = time.time()
    """
"""
print('\ntest done!\n')

# NOTE the avging is buggy for batch_size > 1 as the last batch may be of a different size compared to all other batches
distortion_value_L1_avg = distortion_value_L1_total / len(loader)
distortion_value_MSE_avg = distortion_value_MSE_total / len(loader)
distortion_value_MSSSIM_avg = distortion_value_MSSSIM_total / len(loader)
if not opt.do_not_get_codes:
  code_actual_bpp_avg = code_actual_bpp_total / len(loader)
  code_shannon_bpp_avg = code_shannon_bpp_total / len(loader)
  print('\ntest set avg recon loss (L1/MSE/MS-SSIM) {:.4f}/{:.4f}/{:.4f}, avg pre-/(estimated) post-entropy coding bpp {:.4f}/{:.4f}\n'.format(distortion_value_L1_avg, distortion_value_MSE_avg, distortion_value_MSSSIM_avg, code_actual_bpp_avg, code_shannon_bpp_avg))
else:
  print('\ntest set avg recon loss (L1/MSE/MS-SSIM) {:.4f}/{:.4f}/{:.4f}\n'.format(distortion_value_L1_avg, distortion_value_MSE_avg, distortion_value_MSSSIM_avg))
