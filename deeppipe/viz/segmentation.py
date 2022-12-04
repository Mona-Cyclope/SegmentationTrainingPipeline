import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat

def batch_tensor_to_image_list(batch_tensor):
    image_list = [ rearrange(batch_tensor[i], 'c h w -> h w c').detach().cpu().int().numpy() for i in range(batch_tensor.shape[0]) ]
    return image_list

def batch_tensor_to_mask_list(batch_tensor):
    mask_list = [ batch_tensor[i].detach().cpu().int().numpy() for i in range(batch_tensor.shape[0]) ]
    return mask_list

def show_images_masks(images, masks, alpha=0.3, **kwargs):
    assert len(images) == len(masks), "images and masks must be the same!"
    N = len(images)
    n = int(np.round(np.sqrt(len(images))))
    plus_row = N>n*n
    fig, axs = plt.subplots(ncols=n, nrows=n+plus_row, **kwargs)
    for idx, (image, mask) in enumerate(zip(images, masks)):
        axs[idx//n, idx%n].imshow(image)
        axs[idx//n, idx%n].imshow(mask, alpha=alpha)
    return fig, axs