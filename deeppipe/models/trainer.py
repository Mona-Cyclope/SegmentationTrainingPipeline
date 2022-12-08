import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from deeppipe.viz.segmentation import show_images_masks, batch_tensor_to_image_list, batch_tensor_to_mask_list

import matplotlib as plt
import einops

def binary_dice_loss(inputs, targets, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    inputs = F.sigmoid(inputs)       
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    return 1 - dice

def mc_dice_loss(inputs, targets, n_classes, smooth=1):
    assert n_classes>1, "at least one class"
    dice_loss = 0.0
    for i in range(n_classes):
        bin_y = (targets==i).float()
        bin_pred = inputs[:,i]
        dice_loss += binary_dice_loss(bin_pred, bin_y)
    dice_loss = dice_loss/n_classes
    return dice_loss

class SegmenTrainer(pl.LightningModule):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.save_segmentation_function = None
        self.set_figure_params()
        self.model = None
        self.n_classes = None
        
    def set_figure_params(self, log_fig_params={}):
        self.__log_fig_params = log_fig_params

    def training_step(self, batch, batch_nb):
        x, y = batch
        # mosaic
        if th.rand(1) > 0.5:
            if th.rand(1) > 0.5:
                x = einops.rearrange(x, 'b c (h1 h2) (w1 w2) -> b c (w1 h2) (h1 w2)', h1=2, w1=2)
                y = einops.rearrange(y, 'b (h1 h2) (w1 w2) -> b (w1 h2) (h1 w2)', h1=2, w1=2)
            if th.rand(1) > 0.5:
                b = x.shape[0] - x.shape[0]%2
                x = x[b//2:b]*0.8 + x[:b//2]*0.2
                y = y[b//2:b]
            # const to 0 mapping
            if th.rand(1) > 0.9:
                x = x*0 
                y = y*0
        
        y_hat = self.model(x)
        xent_loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
                    F.binary_cross_entropy_with_logits(y_hat, y)
        dice_loss = mc_dice_loss(y_hat, y, self.n_classes) if self.n_classes > 1 else \
                    binary_dice_loss(y_hat, y)
        loss = (dice_loss + xent_loss)*0.5
        return {'loss': loss, 'xent_loss':  xent_loss, 'dice_loss': dice_loss}
        
    def training_epoch_end(self, outputs):
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        avg_xent_loss = th.stack([x['xent_loss'] for x in outputs]).mean()
        avg_dice_loss = th.stack([x['dice_loss'] for x in outputs]).mean()
        self.log('loss', {'train_loss':avg_loss})

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        xent_loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
            
        dice_loss = mc_dice_loss(y_hat, y, self.n_classes) if self.n_classes > 1 else \
                    binary_dice_loss(y_hat, y)
        
        loss = (dice_loss + xent_loss)*0.5
        
        _,y_hat_hard = y_hat.max(1)
        image_list = batch_tensor_to_image_list(x)
        mask_list = batch_tensor_to_mask_list(y)
        pred_list = batch_tensor_to_mask_list(y_hat_hard)
        try:
            gt_fig, _ = show_images_masks(image_list, mask_list, **self.__log_fig_params)
            self.logger.experiment.add_figure('validation_GT_batch_{}'.format(batch_nb), gt_fig, self.current_epoch)
            pred_fig, _ = show_images_masks(image_list, pred_list, **self.__log_fig_params)
            self.logger.experiment.add_figure('validation_PRED_batch_{}'.format(batch_nb), pred_fig, self.current_epoch)
        except Exception as e:
            print(e)
            print("continue")
        
        #sub_dir = self.logger.sub_dir
        #sub_image_dir = os.path.join(sub_dir, "images", "epoch_{}".format(self.current_epoch))
        #gt_fname = os.path.join(sub_image_dir, 'validation_GT_batch_{}'.format(batch_nb))
        #pred_fname = os.path.join(sub_image_dir, 'validation_PRED_batch_{}'.format(batch_nb))
        #os.makedirs(sub_image_dir, exist_ok=True)
        #plt.savefig(gt_fname, gt_fig)
        #plt.savefig(pred_fname, pred_fig)
        
        return {'loss': loss, 'xent_loss':  xent_loss, 'dice_loss': dice_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        avg_xent_loss = th.stack([x['xent_loss'] for x in outputs]).mean()
        avg_dice_loss = th.stack([x['dice_loss'] for x in outputs]).mean()
        self.log('loss', {'valid_loss':avg_loss})
        return {'avg_val_loss': avg_loss, 'avg_xent_val_loss': avg_xent_loss, 'avg_dice_val_loss': avg_dice_loss }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer