import torch
import numpy as np
import matplotlib.pyplot as plt

def show_images_masks(images, masks, alpha=0.3, **kwargs):
    assert len(images) == len(masks), "images and masks must be the same!"
    N = len(images)
    n = int(np.floor(np.sqrt(len(images))))
    fig, axs = plt.subplots(ncols=n, nrows=n+int(N%n!=0), **kwargs)
    for idx, (image, mask) in enumerate(zip(images, masks)):
        axs[idx//n, idx%n].imshow(image)
        axs[idx//n, idx%n].imshow(mask, alpha=alpha)
    return fig, axs