# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:20:28 2018

@author: feywell
"""

import mxnet as mx
from mxnet import nd
import math
#import matplotlib.pyplot as plt
import numpy as np
irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
 
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = nd.stack(tensor, axis=0)

    if len(tensor.shape) == 2:  # single image H x W
        tensor = tensor.reshape(1, tensor.shape[0], tensor.shape[1])
    if len(tensor.shape) == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = nd.concat(tensor, tensor, tensor, dim=0)
        tensor = tensor.reshape(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])

    if len(tensor.shape) == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = nd.concat(tensor, tensor, tensor, dim=1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clip(a_min=min, a_max=max)
            img.__iadd__(-min).__idiv__(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.shape[0] == 1:
        return tensor.reshape(tensor.shape[1],tensor.shape[2],tensor.shape[3])

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = nd.full(shape=(3, height * ymaps + padding, width * xmaps + padding),val=pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid[:,y * height + padding:(y+1)*height,
                 x * width + padding:(x+1)*width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.asnumpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
    
#if __name__ == "__main__":
#    data = mx.nd.array([[1,2,3],[4,5,6],[7,8,9]])
#    data = data.reshape(1,3,3)
#    data = nd.concat(data, data, data, dim=0)
#    data = nd.expand_dims(data,axis=0)
#    data_in = mx.nd.tile(data,reps =(64,1,1,1))
#    grid = convert_image_np(make_grid(data_in))
#    
#    f, axarr = plt.subplots(1, 2)
#    axarr[0].imshow(grid)
#    axarr[0].set_title('Dataset Images')
    
