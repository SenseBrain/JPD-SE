"""
Modified from:
  https://github.com/NVIDIA/pix2pixHD/blob/master/models/pix2pixHD_model.py
"""

import os
import sys

import numpy as np

import torch
from torch.autograd import Variable

from ctu.utils.image_pool import ImagePool
from ctu.models.pix2pixHD_networks.base_model import BaseModel
from ctu.models.pix2pixHD_networks import networks
from ctu.utils.misc import tensor2im

class Pix2PixHDModel(BaseModel):

  @staticmethod
  def modify_commandline_options(parser, train):
    # architecture
    parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
    parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN. Specifying this flag causes a bug right now')
    parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images for the discriminator')
    parser.add_argument('--no_instance', action='store_true',
        help='If specified, do *not* add instance map as input')
    parser.add_argument('--no_label', action='store_true', help='if specified, do *not* feed the semantic segmentation map to the generator. Should *not* be used together with no_feat. Otherwise, nothing will be fed into the generator and you should expect an exception thrown in your face')
    parser.add_argument('--sem_masking', action='store_true', help='if specified, mask the input image w/ the semantic seg maps')
    parser.add_argument('--binary_mask', action='store_true', help='if specified, fill the semantics regions with ones instead of parts from the original image')
    parser.add_argument('--netE_groups', type=int, default=1, help='depth-wise conv when using semantic masking. Must devide # netE binarizer channel. A simple choice would be = # of semantic channels and set # netE binarizer channel to some integer multiple of # of semantic channels')
    parser.add_argument('--inst_wise_pool', action='store_true', help='if specified, apply instance-wise avg pooling to the output of the visuals encoder. This assumes no_feat_encoding is False')
    # FIXME --norm flag is buggy
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
    parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')

    # objective function
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
    parser.add_argument('--lambda_distortion', type=float, default=10.0, help='weight for reconstruction distortion loss')                
    parser.add_argument('--anneal_lambda', action='store_true', help='if specified, anneal lambda_distortion')
    parser.add_argument('--anneal_interval', type=int, default=5000)
    parser.add_argument('--anneal_factor', type=float, default=5., help='lambda_distortion will be multiplied by this number for every anneal_interval steps')
    parser.add_argument('--match_raw_feat', action='store_true', help='if specified, include the raw features in the discriminator feature matching loss')
    parser.add_argument('--no_gan_feat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
    parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
    parser.add_argument('--no_distortion_loss', action='store_true', help='if specified, do *not* use reconstruction distortion loss')        
    parser.add_argument('--no_g_gan_loss', action='store_true', help='if specified, do *not* use GAN loss for the generator')        
    parser.add_argument('--no_d_gan_loss', action='store_true', help='if specified, do *not* use GAN loss for the discriminator')        

    # data I/O
    parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
    parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--use_compressed', action='store_true', help='if specified, feed the model w/ images compressed by some other codec')
    parser.add_argument('--ext', type=str, default='jpg', choices=['jpg', 'j2k', 'bpg', 'webp'])
    parser.add_argument('--quality', type=str, default='100', help='compression quality for the other codec, only effective when use_compressed is true. Different quality scale apply to different codec, see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for more details. Can control the quality for each semantics channel individually by passing a string of ints separated by commas. This mode is only effective when using sem_masking')
    parser.add_argument('--zero_sem', action='store_true', help='if specified, zero out the semantics. Note that this is different from not using any semantics as in the latter case the models will be defined differently')
    parser.add_argument('--zero_ins', action='store_true', help='if specified, zero out the instance edge map. Note that this is different from not using any instance edge map as in the latter case the models will be defined differently')
    parser.add_argument('--zero_vis', action='store_true', help='if specified, zero out the visuals. Note that this is different from not using any visuals as in the latter case the models will be defined differently')

    # model I/O
    parser.add_argument('--checkpoints_dir', type=str, help='directory in which the trained model checkpoints are saved') 

    # for generator
    parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG. Can be either "global" or "local" (for larger imgs). Note that the local enhancers do not support binarization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
    parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
    parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
    parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
    parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

    # for feature encoding
    parser.add_argument('--no_feat_encoding', action='store_true', help='if specified, directly feed the real image into the generator')
    parser.add_argument('--no_feat', action='store_true', help='if specified, do not use features from the real image')
    parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
    parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
    parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in the first conv layer')        
    parser.add_argument('--use_netE_output', action='store_true', help='if specified, short-circuit the generator and use the visual encoder output as the reconstructed img')

    # for label encoding
    parser.add_argument('--no_label_encoding', action='store_true', help='if specified, do *not* encode the semantic segmentation map being feeding it into the generator')        
    parser.add_argument('--label_encoder_out_channels', type=int, default=36, help='vector length for encoded segmentation map. Default to the case where the label is the one-hot-coded segmentation map (35 channels) concatenated with the instance edge map (1 channel).')        
    parser.add_argument('--n_downsample_E4label', type=int, default=4, help='# of downsampling layers in encoder') 
    parser.add_argument('--ne4lf', type=int, default=64, help='# of label encoder filters in the first conv layer')        

    # for real image E
    parser.add_argument('--no_encoder_binarization', action='store_true', help='if specified, do *not* binarize the encoder bottleneck')
    parser.add_argument('--encoder_binarizer_out_channels', type=int, default=128)

    # for segmentation map E
    parser.add_argument('--no_label_encoder_binarization', action='store_true', help='if specified, do *not* binarize the encoder bottleneck')
    parser.add_argument('--label_encoder_binarizer_out_channels', type=int, default=128)

    # generator binarization
    parser.add_argument('--no_generator_binarization', action='store_true', help='if specified, do *not* binarize the generator bottleneck')
    parser.add_argument('--bin_generator_before_res', action='store_true', help='if specified, binarizer the generator before its res blocks. Only effective when no_generator_binarization is not specified')
    parser.add_argument('--generator_binarizer_out_channels', type=int, default=128)
    return parser


  def __init__(self, opt):
    super(Pix2PixHDModel, self).__init__(opt)
    if ((not opt.no_feat_encoding and not opt.no_encoder_binarization) or (not opt.no_label_encoding and not opt.no_label_encoder_binarization)) and not opt.no_generator_binarization:
      raise ValueError('Usually you only need to binarize the encoders *or* the generator, but you chose to binarize more than what is needed. Is this what you want?')
    # TODO (shiyu) could do with new_zeros, etc.
    self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
    self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
    self.is_train = opt.is_train
    self.use_features = not opt.no_feat 
    if opt.sem_masking: assert self.use_features and not opt.no_label
    
    # figuring out the in_channels of the models
    if not opt.no_label:
      if opt.no_label_encoding:
        # TODO seems that setting num_labels to 0 in the original pix2pixHD would cause the model
        # to use the non-one-hot-coded label map, which does not make sense anyway. So we are 
        # ignoring the possibility of opt.num_labels being 0
        # input_nc = opt.num_labels if opt.num_labels != 0 else opt.input_nc
        semantics_nc = opt.num_labels + 1 if opt.contain_dontcare_label else opt.num_labels
      else:
        semantics_nc = opt.label_encoder_out_channels
    else:
      semantics_nc = 0 

    ##### define networks    
    # Generator network
    netG_input_nc = semantics_nc    
    if opt.no_label_encoding and not opt.no_instance:
      # if encode label, would have encoded the instance edge map together if available,
      # creating a label map w/ opt.label_encoder_out_channels channels. Therefore,
      # there is no need to increase the input channel number by one in that case
      netG_input_nc += 1

    if self.use_features:
      if not opt.no_feat_encoding:
        netG_input_nc += opt.feat_num # semantics + encoded img
      else:
        netG_input_nc += opt.input_nc # semantics + raw img

    if opt.sem_masking:
      if not opt.no_feat_encoding:
        netG_input_nc = opt.feat_num
      else:
        netG_input_nc = opt.input_nc * (opt.num_labels + 1 if not opt.no_instance else opt.num_labels)

    self.netG = networks.define_G(netG_input_nc, opt.num_out_channels, opt.ngf, opt.netG, 
                    opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                    opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids,
                    binarize_generator=not opt.no_generator_binarization, bin_generator_before_res=opt.bin_generator_before_res, generator_binarizer_out_channels=opt.generator_binarizer_out_channels)    

    # Discriminator network
    if self.is_train:
      use_sigmoid = opt.no_lsgan

      netD_input_nc = semantics_nc + opt.num_out_channels # semantics + generated img
      if not opt.no_instance and opt.no_label_encoding:
        netD_input_nc += 1

      # self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
      #               opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
      self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                      opt.num_D, True, gpu_ids=self.gpu_ids)

    ### Encoder network for real image
    if self.use_features and not opt.no_feat_encoding:      
      if opt.sem_masking:
        netE_in = (opt.num_labels + 1) * opt.input_nc if not opt.no_instance else opt.num_labels * opt.input_nc
      else:
        netE_in = opt.input_nc
      self.netE = networks.define_G(netE_in, opt.feat_num, opt.nef, 'encoder', 
                      opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids,
                      binarize_encoder=not opt.no_encoder_binarization,
                      encoder_binarizer_out_channels=opt.encoder_binarizer_out_channels, encoder_groups=opt.netE_groups)  

    ### Encoder network for semantic segmentation map
    if not opt.no_label and not opt.no_label_encoding and not opt.sem_masking:      
      # encode instance edge map together if it is available
      in_channels = opt.num_labels + 1 if not opt.no_instance else opt.num_labels
      self.netE4label = networks.define_G(in_channels, opt.label_encoder_out_channels, opt.ne4lf, 'encoder', 
                      opt.n_downsample_E4label, norm=opt.norm, gpu_ids=self.gpu_ids,
                      binarize_encoder=not opt.no_label_encoder_binarization,
                      encoder_binarizer_out_channels=opt.label_encoder_binarizer_out_channels)  

    print('---------- networks initialized -------------')

    # load networks
    if not self.is_train or opt.load_model:
      self.load_network(self.netG, 'G', opt)        
      if self.is_train:
        self.load_network(self.netD, 'D', opt)  
      if self.use_features and not opt.no_feat_encoding:
        self.load_network(self.netE, 'E', opt)        
      if not opt.no_label and not opt.no_label_encoding and not opt.sem_masking:
        self.load_network(self.netE4label, 'E4label', opt)        

    # set loss functions and optimizers
    if self.is_train:
      if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
        raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
      self.fake_pool = ImagePool(opt.pool_size)

      # define loss functions
      # self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_distortion_loss)
      
      self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
      self.criterionFeat = torch.nn.L1Loss()
      self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        
    
      # Names so we can breakout loss
      # self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_Distortion','D_real', 'D_fake')
      self.loss_names = ('G_GAN','G_GAN_Feat','G_VGG','G_Distortion','D_real', 'D_fake')
    else:
      self.loss_names = ('G_Distortion')
    self.opt = opt
    if self.opt.distortion_loss_fn == 'l1':
      self.criterionDistortion = torch.nn.L1Loss()
    elif self.opt.distortion_loss_fn == 'mse':
      self.criterionDistortion = torch.nn.MSELoss()

  
  @staticmethod
  def init_loss_filter(use_gan_feat_loss, use_vgg_loss, use_distortion_loss):
    flags = (True, use_gan_feat_loss, use_vgg_loss, use_distortion_loss, True, True)
    def loss_filter(g_gan, g_gan_feat, g_vgg, g_distortion, d_real, d_fake):
      return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,g_distortion,d_real,d_fake),flags) if f]
    return loss_filter


  def forward(self, x_dict, opt, mode='get_train_loss'):
    x_dict = self.preprocess(x_dict)  

    if mode == 'get_img':
      return self.get_img(x_dict)
    elif mode == 'get_code':
      return self.get_code(x_dict)
    elif mode == 'get_train_loss':
      return self.get_train_loss(x_dict)
    elif mode == 'get_eval_loss':
      return self.get_eval_loss(x_dict)
    elif mode == 'get_eval_rate':
      return self.get_eval_rate(x_dict)
    else:
      raise ValueError('Invalid forward mode: {}'.format(mode))


  def create_optimizers(self, opt):
    # initialize optimizers
    # optimizer G
    if opt.niter_fix_global > 0:        
      # when training local enhancers, the global generator that has been trained earlier will be 
      # kept fixed for a number of epochs before joint fine-tuning. See the paper for more details
      import sys
      if sys.version_info >= (3,0):
        finetune_list = set()
      else:
        from sets import Set
        finetune_list = Set()

      params_dict = dict(self.netG.named_parameters())
      params = []
      for key, value in params_dict.items():     
        if key.startswith('model' + str(opt.n_local_enhancers)):          
          params += [value]
          finetune_list.add(key.split('.')[0])  
      print('------------- only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
      print('the layers that are finetuned are ', sorted(finetune_list))               
    else:
      params = list(self.netG.parameters())
    if self.use_features and not opt.no_feat_encoding:        
      params += list(self.netE.parameters())       
    if not opt.no_label and not opt.no_label_encoding and not opt.sem_masking:        
      params += list(self.netE4label.parameters())     
    optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))              

    # optimizer D            
    params = list(self.netD.parameters())  
    optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2))
    return optimizer_G, optimizer_D
    

  def use_gpu(self):
    return len(self.opt.gpu_ids) > 0

  
  @staticmethod
  def converter(filename, ext, quality):
    '''
    Convert image to some format in some quality and save it in tmp_folder.

    Params:
      filename: path/to/the/image/to/be/converted

    returns:
      filename: path/to/the/converted/image
    '''
    import os
    import sys
    from PIL import Image
 
    save_name = os.path.splitext(filename)[0] + '.' + ext
    # print('Converting {} into {} w/ quality {}'.format(filename, ext, q))

    if ext in ['jpg', 'webp']:
      img = Image.open(filename)
      img.save(save_name, quality=quality)
      return save_name
    elif ext == 'j2k':
      img = Image.open(filename)
      # TODO support >1 quality layers
      img.save(save_name, quality_mode='rates', quality_layers=[quality])
      return save_name
    elif ext == 'bpg':
      # uses the official BPG encoder as PIL does not currently support BPG
      import subprocess
      # FIXME why is this version buggy?
      # subprocess.run(['bpgenc', '-q '+quality+' -o '+save_name, filename])
      visualization_name = os.path.splitext(save_name)[0] + '_decoded_from_bpg.png'
      subprocess.run('bpgenc -q '+str(quality)+' -o '+save_name + ' ' + filename, shell=True)
      subprocess.run('bpgdec -o '+ visualization_name + ' ' + save_name, shell=True)
      return visualization_name
    else:
      raise ValueError('format must be one of jpg, webp, j2k, or bpg')
  

  def compress(self, x_dict, tmp_folder):
    '''
    Compress the original image using jpg and the use the compressed image as the input image.
    It is important that you specify a different tmp_folder for each running process
    to prevent them from overwritting each other's compressed image.

    Params:
      tmp_folder: a folder to temporarily stash the compressed image
    '''
    import io
    from PIL import Image
    
    import torchvision.transforms as transforms
    from ctu.utils.misc import tensor2im

    # TODO now this code assumes batch size is 1 as PIL's Image can only handle one image at a time
    # TODO compress the pre- or post-preprocessing img, which is better? If the former, should perhaps move this piece of code to dataloader before the preprocessings
    img = Image.fromarray(tensor2im(image_tensor=x_dict['image'], opt=self.opt).squeeze())
    img_name = os.path.join(tmp_folder, 'tmp_image.png')
    # not saving to an actual file but a memory buffer
    if len(self.opt.quality) > 1:
      assert self.opt.sem_masking
    for i, q in enumerate(self.opt.quality):
      
      # compress and decompress using the outside codec
      img.save(img_name)
      compressed_img_name = self.converter(img_name, self.opt.ext, q)
      normalize_fn = transforms.Normalize(self.opt.normalize_mean, self.opt.normalize_std)
      to_tensor = transforms.ToTensor()
      compressed_img = normalize_fn(to_tensor(Image.open(compressed_img_name))).unsqueeze(0)
      # print(compressed_img.size())

      if self.use_gpu():
        compressed_img = compressed_img.cuda()

      all_compressed = compressed_img if i == 0 else torch.cat((all_compressed, compressed_img), dim=1) 
      
    return all_compressed
    

  def preprocess(self, x_dict):         
    
    if self.opt.use_compressed:
      tmp_dir = os.path.join(self.opt.save_dir, 'tmp_imgs')
      if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
      compressed_img = self.compress(x_dict, tmp_dir)

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

      if self.opt.data_type == 16:
        label_tensor = label_tensor.half()
    else:
      label_tensor = None 

    # get edges from instance map
    if not self.opt.no_instance:
      # (shiyu) not feasible to use one-hot repr for instance labels since there are too many classes
      instance_tensor = x_dict['instance']
      # (shiyu) this is much faster on GPU (benchmarked w/ batchsize=1).
      # So do not convert in dataloader
      edge_tensor = self.get_edges(instance_tensor)
      if not self.opt.no_label:
        label_tensor = torch.cat((label_tensor, edge_tensor), dim=1)
      else:
        label_tensor = edge_tensor # (shiyu) does it make more sense to use edge_tensor or instance_tensor or is this case completely useless?

    if self.opt.sem_masking:
      img_tensor = x_dict['image'] if not self.opt.use_compressed else compressed_img
      masked_img = self.sem_mask(img_tensor, label_tensor, self.opt.binary_mask, self.opt.input_nc)

    x_dict = {
        'input_label': label_tensor,
        'real_image': x_dict['image'], 
        'instance_ids': instance_tensor,
        }
    if self.opt.sem_masking:
      x_dict['masked_img'] = masked_img
    if self.opt.use_compressed:
      x_dict['compressed_img'] = compressed_img

    return x_dict
    # return label_tensor, img_tensor, masked_img, instance_tensor
  

  @staticmethod
  def sem_mask(img_tensor, label_tensor, binary_mask=False, img_nc=3):
    if img_tensor.size(1) > img_nc:
      # guessing you've used semantics-aware compression w/ an outside codec
      if img_tensor.size(1) / img_nc != label_tensor.size(1):
        raise ValueError('Either your semantics-aware compression using outside codec is buggy or your img_tensor size is wrong...')
      if binary_mask:
        # each semantics channel is filled w/ 1 in the corresponding semantics regions
        masked_img = img_tensor[:, :img_nc, ...].new_ones(img_tensor[:, :3, ...].size()) * label_tensor[:, 0, ...].unsqueeze(1)
      else:
        # each semantics channel is filled w/ clips from the original image in the corresponding semantics regions
        masked_img = img_tensor[:, :img_nc, ...] * label_tensor[:, 0, ...].unsqueeze(1)
      for i in range(1, label_tensor.size(1)):
        img_slice = img_tensor[:, i * img_nc: (i + 1) * img_nc]
        if binary_mask:
          masked_img = torch.cat((masked_img, img_slice.new_ones(img_slice.size()) * label_tensor[:, i, ...].unsqueeze(1)), 1)
        else:
          masked_img = torch.cat((masked_img, img_slice * label_tensor[:, i, ...].unsqueeze(1)), 1)
    
    else:
      if binary_mask:
        # each semantics channel is filled w/ 1 in the corresponding semantics regions
        masked_img = img_tensor.new_ones(img_tensor.size()) * label_tensor[:, 0, ...].unsqueeze(1)
      else:
        # each semantics channel is filled w/ clips from the original image in the corresponding semantics regions
        masked_img = img_tensor * label_tensor[:, 0, ...].unsqueeze(1)
      for i in range(1, label_tensor.size(1)):
        if binary_mask:
          masked_img = torch.cat((masked_img, img_tensor.new_ones(img_tensor.size()) * label_tensor[:, i, ...].unsqueeze(1)), 1)
        else:
          masked_img = torch.cat((masked_img, img_tensor * label_tensor[:, i, ...].unsqueeze(1)), 1)
      
    return masked_img

    
  def discriminate(self, input_label, test_image, use_pool=False, keep_input=False):
    # this method cuts off the grad graph of input_label and test_image. Use to 
    # obtain loss for the discriminator BUT NOT THE GENERATOR OR THE ENCODERS as these
    # modules need grad from test_image and input_label
    input_concat = torch.cat((input_label.detach(), test_image.detach()), dim=1)
    if use_pool:      
      fake_query = self.fake_pool.query(input_concat)
      return self.netD.forward(fake_query, keep_input)
    else:
      return self.netD.forward(input_concat, keep_input)


  def get_img(self, x_dict):
    with torch.no_grad():
      return self._get_img(x_dict)[0]

  
  def get_eval_rate(self, x_dict):
    with torch.no_grad():
      real_image = x_dict['real_image']
      shannon_bpp_total = 0.
      actual_bpp_total = 0.
      codes = self.get_code(x_dict)
      
      # entropy coding would be performed on each image individually, so we have to
      # iterate through each image in the batch
      for codes_ in codes: # codes is a list
        if codes_ is None: # label_codes is None when sem masking
          continue
        for j in range(real_image.size(0)):
          code = codes_[j] 
          original_img_size = np.prod([*real_image[j].size()[-2:]])
            
          code_p = torch.mean(code)
          code_entropy = - code_p * torch.log(code_p) - (1 - code_p) * torch.log(1 - code_p)
          image_shannon_bpp = code_entropy * code.size(-1) / original_img_size

          shannon_bpp_total += image_shannon_bpp
          actual_bpp_total += (code.size(-1)) / original_img_size 
        
      return shannon_bpp_total / real_image.size(0), \
          actual_bpp_total / real_image.size(0)


  def get_code(self, x_dict):
    """
    Get binary codes of the input image and the label from their respective encoders.

    returns:
      label_code
      image_code
    """

    with torch.no_grad():
      return self._get_img(x_dict, mode='get_binary_code')


  def _get_img(self, x_dict, mode='get_continuous_img'):
    """
    returns:
      fake_image
      input_label
    """
    input_label, real_image, instance_ids = x_dict['input_label'], x_dict['real_image'],  x_dict['instance_ids']
    if self.opt.sem_masking:
      masked_img = x_dict['masked_img']
    if self.opt.use_compressed:
      real_image = x_dict['compressed_img']
      
    """
    # sanity check below
    print(real_image.size())
    import sys

    from PIL import Image
    import skimage.io as io

    from ctu.utils.misc import tensor2im, tensor2label
    real_img = Image.fromarray(tensor2im(x_dict['real_image'][0], self.opt).squeeze())
    real_img.save('real_img.png')
    label = Image.fromarray(tensor2label(x_dict['input_label'][0], 151).squeeze())
    label.save('label.png')
    instance = Image.fromarray(tensor2im(x_dict['instance_ids'][0], self.opt).squeeze())
    instance.save('instance.png')
    sys.exit()
    """
    """
    # sanity check (continued)
    for i in range(masked_img.size(1) // 3):
      masked_img_i = Image.fromarray(tensor2im(masked_img[0][3 * i: 3 * i + 3], self.opt).squeeze())
      masked_img_i.save('masked_img_'+str(i)+'.png') 
    
    sys.exit()
    """
    if self.opt.sem_masking:
      real_image = masked_img
    
    if mode == 'get_binary_code':
      code_list = []

    if not self.opt.sem_masking and not self.opt.no_label_encoding:
      # print('encoding the input semantics...')
      if mode == 'get_continuous_img':
        input_label = self.netE4label(input_label)
      elif mode == 'get_binary_code':
        if not self.opt.no_label_encoder_binarization:
          label_code = self.netE4label(input_label, mode='get_binary_code') if not self.opt.sem_masking else None
          # resize and convert to 0s and 1s
          label_code = (label_code.view(label_code.size(0), -1) + 1) / 2.
          code_list.append(label_code)
      else:
        raise ValueError('Invalid forward mode: {}'.format(mode))

    if self.use_features:
      # print('using features from the real image...')
      if self.opt.no_feat_encoding:
        feat_map = real_image
      else:
        if mode == 'get_continuous_img':
          feat_map = self.netE.forward(real_image, instance_ids, inst_wise_pool=self.opt.inst_wise_pool)             
          if self.opt.use_netE_output:
            return feat_map, input_label
        elif mode == 'get_binary_code':
          if not self.opt.no_encoder_binarization:
            image_code = self.netE.forward(real_image, mode='get_binary_code') if not self.opt.sem_masking else self.netE.forward(masked_img, mode='get_binary_code')            
          # resize and convert to 0s and 1s
          image_code = (image_code.view(image_code.size(0), -1) + 1) / 2.
          code_list.append(image_code)
      
      if mode == 'get_binary_code' and self.opt.no_generator_binarization:
        return code_list

      if self.opt.zero_vis:
        feat_map = feat_map.new_zeros(feat_map.size())
      if self.opt.zero_sem:
        # zero out all semantics
        input_concat = feat_map if self.opt.sem_masking else torch.cat((input_label.new_zeros(input_label.size()), feat_map), dim=1)           
      elif not self.opt.no_instance and self.opt.zero_ins:
        # Only zero out the instance edge map if it was used
        # Note that if the edge map was used, it was concat'ed as the last channel of input_label: see self.preprocess
        input_label[:,-1:,...].mul_(0.)
        input_concat = feat_map if self.opt.sem_masking else torch.cat((input_label, feat_map), dim=1)           

      else:
        input_concat = feat_map if self.opt.sem_masking else torch.cat((input_label, feat_map), dim=1)           
    else:
      if mode == 'get_binary_code' and self.opt.no_generator_binarization:
        return code_list

      if self.opt.zero_sem:
        input_concat = input_label.new_zeros(input_label.size())
      elif not self.opt.no_instance and self.opt.zero_ins:
        input_label[:,-1:,...].mul_(0.)
        input_concat = input_label
      else:
        input_concat = input_label
    
    if mode == 'get_continuous_img':
      fake_image = self.netG.forward(input_concat)
      return fake_image, input_label
    elif mode == 'get_binary_code':
      if not self.opt.no_generator_binarization:
        code = self.netG.forward(input_concat, mode='get_binary_code')
        code = (code.view(code.size(0), -1) + 1) / 2.
        code_list.append(code)
      if len(code_list) > 1:
        print('pix2pixHD: you seem to have binarized more modules than necessary...')
      return code_list

  
  def get_eval_loss(self, x_dict):
    recon_img, input_label = self._get_img(x_dict)

    # print('Semantics-wise distortion (L1):', self.get_sem_wise_distortion(input_label, real_image, fake_image))

    # reconstruction distortion loss
    """
    # visualize images
    import skimage.io as io
    # io.imshow(((real_image * .5 + .5) * 255.).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    io.imshow(((fake_image * .5 + .5) * 255.).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    io.show()
    sys.exit()
    """
    # compute loss after denormalization and quantization to make sure it is accurate
    recon_img = torch.tensor(tensor2im(recon_img, self.opt).transpose(0, 3, 1, 2)).to(torch.float)
    real_img = torch.tensor(tensor2im(x_dict['real_image'], self.opt).transpose(0, 3, 1, 2)).to(torch.float)
    if self.use_gpu():
      recon_img = recon_img.cuda()
      real_img = real_img.cuda()
    loss_G_distortion = self.criterionDistortion(recon_img, real_img)
    
    return loss_G_distortion


  def get_sem_wise_distortion(self, input_label, real_image, fake_image):
    # assumes:
    # input_label.size() = (batch_size, n_classes, h, w)
    # real_image.size() = (batch_size, n_channels, h, w)

    masked_real = self.sem_mask(input_label, real_image, False, self.opt.input_nc)
    masked_fake = self.sem_mask(input_label, fake_image, False, self.opt.input_nc)

    # compute semantics-channel-specific distortion

    # TODO to be made into a unit test for both sem_mask and get_sem_wise_distortion
    """
    real_image = torch.tensor([[[[1, 2.]], [[3, 4]]]])
    fake_image = torch.tensor([[[[0, 2.]], [[9, 11]]]])
    input_label = torch.tensor([[[[1., 1.]], [[0, 0]]]])
    # input_label = torch.tensor([[[[1., 0.]], [[0, 1]]]])
    masked_real = self.sem_mask(input_label, real_image, False, self.opt.input_nc) 
    masked_fake = self.sem_mask(input_label, fake_image, False, self.opt.input_nc) 
    print(real_image.size(), input_label.size(), masked_real.size(), masked_fake.size())
    print(real_image)
    print(fake_image)
    print(input_label)
    print(masked_real)
    print(masked_fake)
    """
    
    # TODO implement MSE
    distortion = torch.abs(masked_real - masked_fake).sum([0, 2, 3]).view(-1, real_image.size(1)).sum(1)
    distortion /= input_label.sum([0, 2, 3]).to(torch.float)

    # if a class doesn't exist in this image, should set the distortion to 0
    # since in this case, we don't need any visual or semantic info
    # into the generator for this class
    distortion[input_label.sum([0, 2, 3]) == 0] = 0
    
    """
    print(distortion.size())
    print(distortion)
    sys.exit()

    # correct output (case 2):
    # torch.Size([1, 2, 1, 2]) torch.Size([1, 2, 1, 2]) torch.Size([1, 4, 1, 2]) torch.Size([1, 4, 1, 2])
    # tensor([[[[1., 2.]], [[3., 4.]]]])
    # tensor([[[[ 0.,  2.]], [[ 9., 11.]]]])
    # tensor([[[[1., 0.]], [[0., 1.]]]])
    # tensor([[[[1., 0.]], [[0., 2.]], [[3., 0.]], [[0., 4.]]]])
    # tensor([[[[ 0.,  0.]], [[ 0.,  2.]], [[ 9.,  0.]], [[ 0., 11.]]]])
    # torch.Size([2])
    # tensor([ 1., 13.])

    # correct output (case 1):
    # torch.Size([1, 2, 1, 2]) torch.Size([1, 2, 1, 2]) torch.Size([1, 4, 1, 2]) torch.Size([1, 4, 1, 2])
    # tensor([[[[1., 2.]], [[3., 4.]]]])
    # tensor([[[[ 0.,  2.]], [[ 9., 11.]]]])
    # tensor([[[[1., 1.]], [[0., 0.]]]])
    # tensor([[[[1., 2.]], [[0., 0.]], [[3., 4.]], [[0., 0.]]]])
    # tensor([[[[ 0.,  2.]], [[ 0.,  0.]], [[ 9., 11.]], [[ 0.,  0.]]]])
    # torch.Size([2])
    # tensor([0.5000, 0.0000])
    """

    # this loss will not be used for bp therefore its grad graph will not be needed anywhere else
    return distortion.detach()


  def get_train_loss(self, x_dict):

    input_label, real_image, instance_ids = x_dict['input_label'], x_dict['real_image'], x_dict['instance_ids']
    fake_image, input_label = self._get_img(x_dict)

    # print('Semantics-wise distortion (L1):', self.get_sem_wise_distortion(input_label, real_image, fake_image))

    # fake detection and loss
    pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
    loss_D_fake = self.criterionGAN(pred_fake_pool, False)      

    # real detection and loss    
    keep_input = True if self.opt.match_raw_feat else False
    pred_real = self.discriminate(input_label, real_image, keep_input=keep_input)
    """
    # visualize that the input is returned properly
    import skimage.io as io
    io.imshow(((pred_real[0][0] * .5 + .5) * 255.).squeeze()[36:].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    io.show()
    sys.exit()
    """
    loss_D_real = self.criterionGAN(pred_real, True)

    # GAN loss (fake passability loss)      
    pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1), keep_input=keep_input)    
    """
    # visualize that the input is returned properly
    import skimage.io as io
    print(len(pred_fake))
    io.imshow(((pred_fake[0][0] * .5 + .5) * 255.).squeeze()[36:].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    print(len(pred_fake[0]))
    io.show()
    sys.exit()
    """
    loss_G_GAN = self.criterionGAN(pred_fake, True)         
    
    # GAN feature matching loss
    loss_G_GAN_Feat = 0.
    # feat_weights = 4.0 / (self.opt.n_layers_D + 1)
    feat_weights = 1.0
    D_weights = 1.0 / self.opt.num_D
    for i in range(self.opt.num_D):
      for j in range(len(pred_fake[i])-1):
        loss_G_GAN_Feat += D_weights * feat_weights * \
          self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
           
    # VGG feature matching loss
    loss_G_VGG = self.criterionVGG(fake_image, real_image) 

    # reconstruction distortion loss
    """
    # visualize images
    import skimage.io as io
    # io.imshow(((real_image * .5 + .5) * 255.).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    io.imshow(((fake_image * .5 + .5) * 255.).squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    io.show()
    sys.exit()
    """
    loss_G_distortion = self.criterionDistortion(fake_image, real_image)
    
    # only return the fake_B image if necessary to save BW
    # return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_distortion, loss_D_real, loss_D_fake ), None if not infer else fake_image ]
    return loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_distortion, loss_D_real, loss_D_fake


  def get_edges(self, t):
    edge = self.ByteTensor(t.size()).zero_()
    edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    if self.opt.data_type==16:
      return edge.half()
    else:
      return edge.float()


  def save(self):
    self.save_network(self.netG, 'G', self.opt) 
    self.save_network(self.netD, 'D', self.opt) 
    if self.use_features and not self.opt.no_feat_encoding:
      self.save_network(self.netE, 'E', self.opt) 
    if not self.opt.no_label and not self.opt.no_label_encoding and not self.opt.sem_masking:
      self.save_network(self.netE4label, 'E4label', self.opt) 


  def update_fixed_params(self, optimizer_G):
    # after fixing the global generator for a number of iterations, also start finetuning it
    params = list(self.netG.parameters())
    if self.use_features and not self.opt.no_feat_encoding:
      params += list(self.netE.parameters())       
    if not self.opt.no_label and not self.opt.no_label_encoding and not self.opt.sem_masking:        
      params += list(self.netE4label.parameters())     
    optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
    if self.opt.verbose:
      print('------------ now also finetuning global generator -----------')
