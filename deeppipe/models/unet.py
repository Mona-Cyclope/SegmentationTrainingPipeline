import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from deeppipe.viz.segmentation import show_images_masks, batch_tensor_to_image_list, batch_tensor_to_mask_list
import matplotlib as plt

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, rootchan=32, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, rootchan*2**(0))
        self.down1 = Down(rootchan*2**(0), rootchan*2**(1))
        self.down2 = Down(rootchan*2**(1), rootchan*2**(2))
        self.down3 = Down(rootchan*2**(2), rootchan*2**(3))
        factor = 2 if bilinear else 1
        self.down4 = Down(rootchan*2**(3), rootchan*2**(4) // factor)
        self.up1 = Up(rootchan*2**(4), rootchan*2**(3) // factor, bilinear)
        self.up2 = Up(rootchan*2**(3), rootchan*2**(2) // factor, bilinear)
        self.up3 = Up(rootchan*2**(2), rootchan*2**(1) // factor, bilinear)
        self.up4 = Up(rootchan*2**(1), rootchan*2**(0), bilinear)
        self.outc = OutConv(rootchan*2**(0), n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
#PyTorch

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

class UnetTrainer(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(*args, **kwargs).cuda()
        self.n_classes = self.model.n_classes 
        self.save_segmentation_function = None
        self.set_figure_params()
        
    def set_figure_params(self, log_fig_params={}):
        self.__log_fig_params = log_fig_params

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        xent_loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
                    F.binary_cross_entropy_with_logits(y_hat, y)
        dice_loss = mc_dice_loss(y_hat, y, self.n_classes) if self.n_classes > 1 else \
                    binary_dice_loss(y_hat, y)
        loss = (dice_loss + xent_loss)*0.5
        return {'loss': loss, 'xent_loss':  xent_loss, 'dice_loss': dice_loss}
        
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_xent_loss = torch.stack([x['xent_loss'] for x in outputs]).mean()
        avg_dice_loss = torch.stack([x['dice_loss'] for x in outputs]).mean()
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
        gt_fig, _ = show_images_masks(image_list, mask_list, **self.__log_fig_params)
        pred_fig, _ = show_images_masks(image_list, pred_list, **self.__log_fig_params)
        self.logger.experiment.add_figure('validation_GT_batch_{}'.format(batch_nb), gt_fig, self.current_epoch)
        self.logger.experiment.add_figure('validation_PRED_batch_{}'.format(batch_nb), pred_fig, self.current_epoch)
        
        #sub_dir = self.logger.sub_dir
        #sub_image_dir = os.path.join(sub_dir, "images", "epoch_{}".format(self.current_epoch))
        #gt_fname = os.path.join(sub_image_dir, 'validation_GT_batch_{}'.format(batch_nb))
        #pred_fname = os.path.join(sub_image_dir, 'validation_PRED_batch_{}'.format(batch_nb))
        #os.makedirs(sub_image_dir, exist_ok=True)
        #plt.savefig(gt_fname, gt_fig)
        #plt.savefig(pred_fname, pred_fig)
        
        return {'loss': loss, 'xent_loss':  xent_loss, 'dice_loss': dice_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_xent_loss = torch.stack([x['xent_loss'] for x in outputs]).mean()
        avg_dice_loss = torch.stack([x['dice_loss'] for x in outputs]).mean()
        self.log('loss', {'valid_loss':avg_loss})
        return {'avg_val_loss': avg_loss, 'avg_xent_val_loss': avg_xent_loss, 'avg_dice_val_loss': avg_dice_loss }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
