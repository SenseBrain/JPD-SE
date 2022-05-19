JPD-SE: High-Level Semantics for Joint Perception-Distortion Enhancement in Image Compression

# Dependencies

- python=3.7
- numpy=1.16.4
- torch=1.1.0
- torchvision=0.3.0
- scikit-image=0.15.0

If you want to use tensorboard to log your training, you will additionally need:

- tensorboard=1.14.0
- tensorflow=1.14.0

If you want to get the MS-SSIM values when testing, you will need to install [this package](https://github.com/jorge-pessoa/pytorch-msssim), which you can do by ```cd /path/to/JPD-SE/; git clone https://github.com/jorge-pessoa/pytorch-msssim.git; cd pytorch_msssim; pip install -e .```, assuming that you'd like to install via ```pip```.

**NOTE**

The main functions *will break* if you use ```torch<=0.4.1``` but should work just fine otherwise.

Although, we have only run the tests under this exact setting so we cannot guarantee that everything would work as expected in a different set-up.

# Installation

`cd` to the root directory of this repo, then

- dev mode: `pip install -e .`
- normal mode: `pip install .`

# TODO
* Share pre-trained phase 1 model pix_novis_nodistortion_newnormalization_1024
* Share pre-trained BPG test models