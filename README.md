# JPD-SE: High-Level Semantics for Joint Perception-Distortion Enhancement in Image Compression
This repo contains the code to perform and evaluate the nerual image compression method introduced in the folloing paper.
> Duan, Shiyu, Huaijin Chen, and Jinwei Gu. "JPD-SE: High-Level Semantics for Joint Perception-Distortion Enhancement in Image Compression." *IEEE Transactions on Image Processing* 31 (2022): 4405-4416.
[[pdf]](https://ieeexplore.ieee.org/iel7/83/4358840/09807639.pdf) [[arxiv]](https://arxiv.org/abs/2005.12810)

Some results are provided below, where the "-SE" codecs, i.e., our semantically enhanced models, outperform the originals.

![img1](figures/img1.png)

**Abstract**
>While humans can effortlessly transform complex visual scenes into simple words and the other way around by leveraging their high-level understanding of the content, conventional or the more recent learned image compression codecs do not seem to utilize the semantic meanings of visual content to their full potential. Moreover, they focus mostly on rate-distortion and tend to underperform in perception quality especially in low bitrate regime, and often disregard the performance of downstream computer vision algorithms, which is a fast-growing consumer group of compressed images in addition to human viewers. In this paper, we (1) present a generic framework that can enable any image codec to leverage high-level semantics and (2) study the joint optimization of perception quality and distortion. Our idea is that given *any codec*, we utilize high-level semantics to augment the low-level visual features extracted by it and produce essentially a new, semantic-aware codec. We propose a three-phase training scheme that teaches semantic-aware codecs to leverage the power of semantic to jointly optimize rate-perception-distortion (R-PD) performance. As an additional benefit, semantic-aware codecs also boost the performance of downstream computer vision algorithms. To validate our claim, we perform extensive empirical evaluations and provide both quantitative and qualitative results.

## Dependencies

- python=3.7
- numpy=1.16.4
- torch=1.1.0
- torchvision=0.3.0
- scikit-image=0.15.0
- [libbpg](https://bellard.org/bpg/)

If you want to use tensorboard to log your training, you will additionally need:

- tensorboard=1.14.0
- tensorflow=1.14.0

If you want to get the MS-SSIM values when testing, you will need to install [this package](https://github.com/jorge-pessoa/pytorch-msssim), which you can do by ```cd /path/to/JPD-SE/; git clone https://github.com/jorge-pessoa/pytorch-msssim.git; cd pytorch_msssim; pip install -e .```, assuming that you'd like to install via ```pip```.

**NOTE**

The main functions *will break* if you use ```torch<=0.4.1``` but should work just fine otherwise.

Although, we have only run tests under this exact setting so we cannot guarantee that everything would work as expected in a different set-up.

## Installation

`cd` to the root directory of this repo, then

- dev mode: `pip install -e .`
- normal mode: `pip install .`

## Train/Test

We provide example scripts for training and testing BPG-based models in [scripts/](scripts/).

## Pre-Trained Models

We provide several pre-trained BPG-based models with quality factors 33, 36, 39, and 42 [here](https://drive.google.com/drive/folders/1qUEU78ZggAG-oSGVQszszIsnYyDjlMNg?usp=sharing). 

The Cityscapes test data used in the paper is provided in [datasets/](datasets/). 
