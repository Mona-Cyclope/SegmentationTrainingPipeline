
import torch as th
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
import numpy as np
import os

from deeppipe.datasets.augmentations import ApplyAlbumination, compose, transposeImage, convertImage2float, convertMask2long
from deeppipe.datasets.io import labeler_image_mask_load, rescale_labeler_image_mask_load, labeler_download_lot
from deeppipe.datasets.datasets import ImageMaskDataset

from deeppipe.configs import segmentation

# collect configs
label_dict = segmentation.label_dict
resize_height_image = segmentation.resize_height_image
resize_width_image = segmentation.resize_height_image
channels_image = segmentation.channels_image
train_batch_folder = segmentation.train_batch_folder
train_batches_json = segmentation.train_batches_json
valid_batch_folder = segmentation.valid_batch_folder
valid_batches_json = segmentation.valid_batches_json
download_images = segmentation.download_images
max_epochs = segmentation.max_epochs
devices = segmentation.devices
accelerator = segmentation.accelerator
log_every_n_steps = segmentation.log_every_n_steps
train_batch_size = segmentation.train_batch_size
valid_batch_size = segmentation.valid_batch_size
albumentation = segmentation.albumentation
model = segmentation.model

# set manula seed
th.manual_seed(42)

# data formatting
# data loading from labeler format
image_mask_load_fun=labeler_image_mask_load(maks_label_dict=label_dict)
# data loadinf and rescale
image_load_fun = rescale_labeler_image_mask_load((resize_height_image, resize_width_image), image_mask_load_fun=image_mask_load_fun)

augmentation = ApplyAlbumination(albumentation)
                                 
# data augmentation and formatting
image_mask_train_transform = compose( [ augmentation, transposeImage, convertImage2float, convertMask2long ] )
# data formatting
image_mask_valid_transform = compose( [ transposeImage, convertImage2float, convertMask2long ] )

# load train data batches
train_batches = []
for train_batch_json in train_batches_json:
    image_folder, mask_folder  = labeler_download_lot(os.path.join(train_batch_folder, train_batch_json), only_labels=not download_images)
    ds = ImageMaskDataset(image_folder, mask_folder, image_prefix="", mask_prefix="", image_suffix=".png", mask_suffix=".json",
                          image_mask_load_fun=image_load_fun, image_mask_transform=image_mask_train_transform)
    train_batches += [ds]
train_dataset = ConcatDataset(train_batches)
    
# load valid data batches
valid_batches = []
for valid_batch_json in valid_batches_json:
    image_folder, mask_folder  = labeler_download_lot(os.path.join(valid_batch_folder, valid_batch_json), only_labels=not download_images)
    ds = ImageMaskDataset(image_folder, mask_folder, image_prefix="", mask_prefix="", image_suffix=".png", mask_suffix=".json",
                          image_mask_load_fun=image_load_fun, image_mask_transform=image_mask_valid_transform)
    valid_batches += [ds]
valid_dataset = ConcatDataset(valid_batches)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

# create model and trainer
trainer = Trainer( max_epochs=max_epochs, accelerator=accelerator, devices=devices, logger=True, log_every_n_steps=log_every_n_steps)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
