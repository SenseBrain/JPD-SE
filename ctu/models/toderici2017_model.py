import sys
import time

import skimage.io as io

import torch

from ctu.models.toderici2017_networks import *
from ctu.quantizers.binarize import *
from ctu.utils.misc import tensor2im


class Toderici2017Model(torch.nn.Module):
  @staticmethod
  def modify_commandline_options(parser, train):
    parser.add_argument('--bin_channels', type=int, default=128,
        help='The number of out_channels for the binarizer.')
    parser.add_argument('--num_iterations', type=int, default=10,
        help='The number of iterations to compress an image.')
    parser.add_argument('--semantics_mode', type=str, default='stack', choices=['none', 'only', 'stack', 'mask'],
        help='How to feed semantics into the codec. "none" means using no semantics; "stack" means concatenating the semantics after the image along the channel axis; "only" means using semantics *only*')
    parser.add_argument('--no_label', action='store_true',
        help='If specified, do *not* add label map as input')
    # for instance-wise features
    parser.add_argument('--no_instance', action='store_true',
        help='If specified, do *not* add instance map as input')
    parser.add_argument('--checkpoint_file', type=str, 
        help='The checkpoint file to load from')
    return parser


  def __init__(self, opt):
    super(Toderici2017Model, self).__init__()
    self.opt = opt
    self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
    self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

    num_img_channels = 3 if opt.semantics_mode != 'only' else 0

    if not opt.no_label:
      if opt.contain_dontcare_label:
        num_label_channels = opt.num_labels + 1
      else:
        num_label_channels = opt.num_labels
    else:
      num_label_channels = 0

    if not opt.no_instance:
      num_ins_channels = 1
    else:
      num_ins_channels = 0

    if opt.semantics_mode != 'mask':
      encoder_in_channels = num_img_channels + num_label_channels + num_ins_channels 
    else:
      encoder_in_channels = num_img_channels + (num_label_channels + num_ins_channels) * 3 


    self.encoder = EncoderCell(encoder_in_channels)
    self.binarizer = Binarizer(512, opt.bin_channels)
    self.decoder = DecoderCell(opt.bin_channels, opt.num_out_channels)
    
    if opt.distortion_loss_fn == 'l1':
      self.loss_fn = torch.nn.L1Loss()
    elif self.opt.distortion_loss_fn == 'mse':
      self.loss_fn = torch.nn.MSELoss()


  def use_gpu(self):
    return len(self.opt.gpu_ids) > 0


  def get_edges(self, t):
    edge = self.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


  def preprocess(self, x_dict, states):
    """
    returns:
      semantics
      img
      states
    """
    # move to GPU and change data types

    if self.use_gpu():
      x_dict['label'] = x_dict['label'].cuda()
      x_dict['instance'] = x_dict['instance'].cuda()
      x_dict['image'] = x_dict['image'].cuda()
    
    # preprocess label map if it exists
    if not self.opt.no_label:
      # (shiyu) this is much faster on GPU (benchmarked w/ batchsize=1).
      # So do not convert in dataloader
      label_map = x_dict['label'].long()
      batch_size, _, h, w = label_map.size()
      num_channels = self.opt.num_labels + 1 if self.opt.contain_dontcare_label \
            else self.opt.num_labels
      input_label = self.FloatTensor(batch_size, num_channels, h, w).zero_()
      label_tensor = input_label.scatter_(1, label_map, 1.0)
      
    # concatenate instance map if it exists
    if not self.opt.no_instance:
      # (shiyu) not feasible to use one-hot repr for instance labels since there are too many classes
      instance_tensor = x_dict['instance']
      # (shiyu) this is much faster on GPU (benchmarked w/ batchsize=1).
      # So do not convert in dataloader
      instance_tensor = self.get_edges(instance_tensor)

    if self.opt.semantics_mode == 'none':
      semantics = None
    else:
      if self.opt.no_instance:
        semantics = label_tensor
      elif self.opt.no_label:
        semantics = instance_tensor
      else:
        semantics = torch.cat((label_tensor, instance_tensor), dim=1)
    
    if self.opt.semantics_mode == 'mask':
      img_tensor = x_dict['image']
      # mask image w/ semantics      
      # uncomment to test w/ batch_size = 2
      # print(label_tensor[:, 0, ...].size())
      masked_img = img_tensor * semantics[:, 0, ...].unsqueeze(1)
      # import skimage.io as io
      # import sys
      # io.imsave('original.jpg', (img_tensor[1] * 255.).cpu().numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8))
      # io.imsave('masked_img_0.jpg', (masked_img[1] * 255.).cpu().numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8))
      for i in range(1, semantics.size(1)):
        # io.imsave('masked_img_'+str(i)+'.jpg', (img_tensor * semantics[:, i, ...].unsqueeze(1) * 255.)[1].cpu().numpy().squeeze().transpose(1, 2, 0).astype(np.uint8))
        masked_img = torch.cat((masked_img, img_tensor * semantics[:, i, ...].unsqueeze(1)), 1)

    if self.opt.semantics_mode in ['stack', 'only']:
      img = torch.cat((x_dict['image'], semantics), dim=1)
    elif self.opt.semantics_mode == 'mask':
      img = torch.cat((x_dict['image'], masked_img), dim=1)
    else:
      img = x_dict['image']

    return semantics, img, states


  def forward(self, x_dict, states, opt, mode='get_train_loss'):
    """
    A wrapper to enable using multi-gpu.
    
    params:
      mode (optional): str
        Choices: 'get_img', 'get_train_loss', 'get_eval_loss'
        get_train_loss: get the training loss term which the optimizer wants to minimize;
        get_eval_loss: get the eval reconstruction loss. For toderici2017, this is not equal to the train loss;
        get_img: get the reconstructed image
    """
    semantics, img, states = self.preprocess(x_dict, states)
    del semantics
    if mode=='get_img':
      img = self.get_img(img, states, opt, mode='get_img')
      return img
    if mode=='get_train_loss':
      loss = self.get_train_loss(x=img, states=states, opt=opt)  
      return loss
    if mode=='get_eval_loss':
      loss = self.get_eval_loss(x=img, states=states, opt=opt)  
      return loss
    if mode=='get_code':
      return self.get_img(x=img, states=states, opt=opt, mode='get_code')
    if mode=='get_eval_rate':
      return self.get_eval_rate(x=img, states=states, opt=opt)
    else:
      raise ValueError('Invalid forward mode: {}'.format(mode))
  

  def create_optimizers(self, opt):
    optimizer = torch.optim.Adam(
        self.parameters(),
        lr=opt.lr,
        betas=(opt.beta1, opt.beta2)
        )
    return optimizer
  

  @staticmethod  
  def get_init_states(opt, height, width):
    return init_lstm(opt.batch_size, height=height, width=width)


  def get_train_loss(self, x, states, opt):

    (encoder_h_1, encoder_h_2, encoder_h_3, 
    decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = states

    # use
    encoder_h_1 = (encoder_h_1[0].to(x.device), encoder_h_1[1].to(x.device))
    encoder_h_2 = (encoder_h_2[0].to(x.device), encoder_h_2[1].to(x.device))
    encoder_h_3 = (encoder_h_3[0].to(x.device), encoder_h_3[1].to(x.device))
    decoder_h_1 = (decoder_h_1[0].to(x.device), decoder_h_1[1].to(x.device))
    decoder_h_2 = (decoder_h_2[0].to(x.device), decoder_h_2[1].to(x.device))
    decoder_h_3 = (decoder_h_3[0].to(x.device), decoder_h_3[1].to(x.device))
    decoder_h_4 = (decoder_h_4[0].to(x.device), decoder_h_4[1].to(x.device))

    res = x
    """
    # sanity check for sem masking below
    import skimage.io as io
    import sys
    for i in range(res.size(1) // 3):
      io.imsave('masked_img_'+str(i)+'.jpg', (res[0][3 * i: 3 * i + 3] * 255.).cpu().numpy().squeeze().transpose((1, 2, 0)).astype(np.uint8))
    
    sys.exit()
    """
    losses = []

    for _ in range(opt.num_iterations):
      if opt.semantics_mode == 'only':
        res_in = res[:, 3:, :, :]
      else:
        res_in = res
      encoded, encoder_h_1, encoder_h_2, encoder_h_3 = self.encoder(
          res_in, encoder_h_1, encoder_h_2, encoder_h_3)
      codes = self.binarizer(encoded)
      output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
          codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
      
      if opt.semantics_mode != 'none':
        placeholder = res[:, 3:, :, :].new_zeros(res[:, 3:, :, :].size())
        output = torch.cat((output, placeholder), dim=1)

      losses.append(self.loss_fn(
        res[:, :3, :, :],
        output[:, :3, :, :]
        ))

      res = res - output
      
    loss = sum(losses) / opt.num_iterations
    return loss

  
  def get_eval_rate(self, x, states, opt):
    with torch.no_grad():
      shannon_bpp_total = 0.
      actual_bpp_total = 0.
      codes = self.get_img(x, states, opt, mode='get_code')

      # entropy coding would be performed on each image individually, so we have to
      # iterate through each image in the batch

      # x should be of dim 4
      for j in range(x.size(0)):
        img_shannon_bpp = 0.
        img_actual_bpp = 0.
        original_img_size = np.prod([*x[j].size()[-2:]])

        for code in codes:
          # codes is a list (w/ len opt.num_iterations) of tensors, each being the binary codes
          # of the residuals of the batch. Entropy coding each residual individually usually
          # produces a lower avg bpp
          res_code = code[j]
          
          res_code_p = torch.mean(res_code)
          res_code_entropy = - res_code_p * torch.log(res_code_p) - (1 - res_code_p) * torch.log(1 - res_code_p)
          res_shannon_bpp = res_code_entropy * res_code.size(-1) / original_img_size
          
          # since we need all the residuals to reconstruct the image, we need to transmit all the res_code
          img_shannon_bpp += res_shannon_bpp
          img_actual_bpp += res_code.size(-1) / original_img_size

        shannon_bpp_total += img_shannon_bpp
        actual_bpp_total += img_actual_bpp 

      return  shannon_bpp_total / x.size(0), \
          actual_bpp_total / x.size(0)


  def get_img(self, x, states, opt, mode='get_img'):
    with torch.no_grad():
      (encoder_h_1, encoder_h_2, encoder_h_3, 
      decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = states

      # use
      encoder_h_1 = (encoder_h_1[0].to(x.device), encoder_h_1[1].to(x.device))
      encoder_h_2 = (encoder_h_2[0].to(x.device), encoder_h_2[1].to(x.device))
      encoder_h_3 = (encoder_h_3[0].to(x.device), encoder_h_3[1].to(x.device))
      decoder_h_1 = (decoder_h_1[0].to(x.device), decoder_h_1[1].to(x.device))
      decoder_h_2 = (decoder_h_2[0].to(x.device), decoder_h_2[1].to(x.device))
      decoder_h_3 = (decoder_h_3[0].to(x.device), decoder_h_3[1].to(x.device))
      decoder_h_4 = (decoder_h_4[0].to(x.device), decoder_h_4[1].to(x.device))

      res = x

      # NOTE in the original implementation, + 0.5 because the last layer is a tanh layer. But can compensate this via normalizing data in preprocessing
      # w/ mean [.5, .5, .5] and std [.5, .5, .5]
      output_img = x.new_zeros(x[:, :3, :, :].size())
      # output_img = x.new_zeros(x[:, :3, :, :].size()) + .5
      output_img.requires_grad = False
      all_codes = []
      for i in range(opt.num_iterations):
        if opt.semantics_mode == 'only':
          res_in = res[:, 3:, :, :]
        else:
          res_in = res
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = self.encoder(
            res_in, encoder_h_1, encoder_h_2, encoder_h_3)
        codes = self.binarizer(encoded)

        all_codes.append((codes.view(codes.size(0), -1) + 1) / 2.)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = self.decoder(
            codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
        
        output_img = output_img + output
        if opt.semantics_mode != 'none':
          placeholder = res[:, 3:, :, :].new_zeros(res[:, 3:, :, :].size())
          output = torch.cat((output, placeholder), dim=1)
        res = res - output
      
      if mode == 'get_code':
        return all_codes
      elif mode == 'get_img':
        return output_img
      else:
        raise ValueError('Invalid mode {}'.format(mode))

  def get_eval_loss(self, x, states, opt):
    with torch.no_grad():
      recon_img = self.get_img(x, states, opt)
      recon_img = torch.tensor(tensor2im(recon_img, self.opt).transpose(0, 3, 1, 2)).to(torch.float)
      real_img = torch.tensor(tensor2im(x[:, :3, :, :], self.opt).transpose(0, 3, 1, 2)).to(torch.float)

      if self.use_gpu():
        recon_img = recon_img.cuda()
        real_img = real_img.cuda()
       
      loss = self.loss_fn(recon_img, real_img)
      return loss.item()
