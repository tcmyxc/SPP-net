# -*- coding: utf-8 -*-

"""
description: 空间金字塔池化层
参考：
1. https://github.com/mmmmmmiracle/SPPNet/blob/master/sppnet.py
2. https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py

"""
import torch
import torch.nn as nn


def spatial_pyramid_pooling(previous_conv, out_pool_size):
    """
    spatial pyramid pooling

    previous_conv: a tensor of previous convolution layer
    out_pool_size: a tuple of expected output size of max pooling layer, must start from 1

    returns: a tensor with shape [1 x n] is the concentration of multi-level pooling
    """
    num_sample = previous_conv.shape[0]  # batch size
    for i in out_pool_size:
        max_pool = nn.AdaptiveMaxPool2d(i)
        x = max_pool(previous_conv)
        if i == 1:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


if __name__ == '__main__':
    previous_conv, out_pool_size = torch.rand((1, 512, 13, 13)), (1, 2, 4)
    print(f"previous_conv shape: {previous_conv.shape}")
    print("spp out", spatial_pyramid_pooling(previous_conv, out_pool_size).shape)