from deeppipe.configs import segmentation_kitti


import torch as th
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import transforms
from pytorch_lightning import Trainer
import numpy as np
import os
from pytorch_lightning.loggers import TensorBoardLogger

from deeppipe.datasets.augmentations import ApplyAlbumination, compose, transposeImage, convertImage2float, convertMask2long
from deeppipe.datasets.io import labeler_image_mask_load, rescale_labeler_image_mask_load, labeler_download_lot
from deeppipe.datasets.datasets import KittiDataset

from deeppipe.configs import segmentation

# collect configs
zip_path = segmentation_kitti.zip_path
zip_ext = segmentation_kitti.zip_ext
max_epochs = segmentation_kitti.max_epochs
devices = segmentation_kitti.devices
accelerator = segmentation_kitti.accelerator
log_every_n_steps = segmentation_kitti.log_every_n_steps
check_val_every_n_epoch = segmentation_kitti.check_val_every_n_epoch
train_batch_size = segmentation_kitti.train_batch_size
valid_batch_size = segmentation_kitti.valid_batch_size
albumentation = segmentation_kitti.albumentation
model = segmentation_kitti.model
logging_folder = segmentation_kitti.logging_folder
model_name = segmentation_kitti.model_name

# set manula seed
th.manual_seed(42)

# data augmentation and formatting
augmentation = ApplyAlbumination(albumentation)
image_mask_train_transform = compose( [ augmentation, transposeImage, convertImage2float, convertMask2long ] )
image_mask_valid_transform = compose( [ transposeImage, convertImage2float, convertMask2long ] )
dataset = KittiDataset(zip_path=zip_path, zip_ext=zip_ext)
train_dataset, valid_dataset = random_split(dataset, [0.9,0.1], generator=th.Generator().manual_seed(42))
train_dataset.dataset.transform = image_mask_train_transform
valid_dataset.dataset.transform = image_mask_valid_transform
# data loaders
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

# create model and trainer
logger = TensorBoardLogger(logging_folder, name=model_name)
trainer = Trainer( max_epochs=max_epochs, accelerator=accelerator, devices=devices, logger=logger, check_val_every_n_epoch=check_val_every_n_epoch, log_every_n_steps=log_every_n_steps)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
