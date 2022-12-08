from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch as th
from deeppipe.configs import segmentation_DAI as segmentation_config
from pytorch_lightning import Trainer
import numpy as np
import os
from pytorch_lightning.loggers import TensorBoardLogger
from deeppipe.datasets.augmentations import ApplyAlbumination
from deeppipe.datasets.datasets import ImageMaskDataset

# dataloader config
batches = segmentation_config.batches
image_load_fun = segmentation_config.image_load_fun
mask_load_fun = segmentation_config.mask_load_fun
image_prefix = segmentation_config.image_prefix
image_mask_preproc_fun = segmentation_config.image_mask_preproc_fun
mask_prefix = segmentation_config.mask_prefix
image_suffix= segmentation_config.image_suffix
mask_suffix= segmentation_config.mask_suffix
train_batch_size = segmentation_config.train_batch_size
valid_batch_size = segmentation_config.valid_batch_size
train_albumentation = segmentation_config.train_albumentation

# training config
logging_folder = segmentation_config.logging_folder
max_epochs = segmentation_config.max_epochs
accelerator = segmentation_config.accelerator
devices = segmentation_config.devices
check_val_every_n_epoch = segmentation_config.check_val_every_n_epoch
log_every_n_steps = segmentation_config.log_every_n_steps
model = segmentation_config.model
model_name = segmentation_config.model_name

# data sets
batches_dataset = []
for batch_id, batch in batches.items():
    image_folder, mask_folder = batch['image_folder'], batch['mask_folder']
    batches_dataset += [ ImageMaskDataset(image_folder, mask_folder, image_load_fun, mask_load_fun,
                                        image_mask_preproc_fun=image_mask_preproc_fun, image_mask_transform=None,
                                        image_prefix=image_prefix, mask_prefix=mask_prefix, image_suffix=image_suffix, mask_suffix=mask_suffix
                                        ) ]
dataset = ConcatDataset(batches_dataset)
train_dataset, valid_dataset = random_split(dataset, [0.9,0.1], generator=th.Generator().manual_seed(42))
train_dataset.dataset.transform = ApplyAlbumination(train_albumentation)

# data loaders
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

# trainer
logger = TensorBoardLogger(logging_folder, name=model_name)
trainer = Trainer( max_epochs=max_epochs, accelerator=accelerator, devices=devices, logger=logger, check_val_every_n_epoch=check_val_every_n_epoch, log_every_n_steps=log_every_n_steps)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
