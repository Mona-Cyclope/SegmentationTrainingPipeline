import numpy as np
import json
import cv2
import os
import wget
from tqdm import tqdm
import pandas
import os
import json
from deeppipe import LOG
import torch
import torch as th
import torchvision.transforms as transforms

def compose(pps):
    def apply(image=None, mask=None):
        for pp in pps:
            image, mask = pp(image=image, mask=mask)
        return image, mask
    return apply

def convertImage2float(image=None, mask=None): return image.astype(np.float32), mask

def convertMask2long(image=None, mask=None): return image, mask.astype(np.int64)

def transposeImage(image=None, mask=None): return np.transpose(image, [2,0,1]), mask

def convertImageRgb2Hsv(image=None, mask=None): return cv2.cvtColor(image, cv2.COLOR_BGR2HSV), mask

def convertImageRgb2Gry(image=None, mask=None): return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask

def Numpy2Tensor():
    def f(image=None, mask=None):
        return torch.from_numpy(image), torch.from_numpy(mask)
    return f

def normCeneterImage(mean, var):
    def f(image=None, mask=None, mean=mean, var=var): 
        return (image-mean)/var, mask
    return f

def preprocVGG():
    pp = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    def f(image=None, mask=None, pp=pp):
        image = pp(image)
        return image, mask
    return f

def image_mask_resize(dim):
    def f(image=None, mask=None):
        image = cv2.resize(image, tuple(dim), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, tuple(dim), interpolation=cv2.INTER_NEAREST)
        return image, mask
    return f

def get_mask_from_coordinates(coordinates, shape, value=1, mask=None):
    H,W = shape
    mask = np.zeros([H,W], dtype=np.uint8) if mask is None else mask
    hull = np.expand_dims(np.asarray(coordinates), axis=1)
    cv2.drawContours(mask, [hull], -1, value, -1)
    return mask

def get_coordinates(all_points_x, all_points_y):
    coordinates = []
    for x, y in zip(all_points_x, all_points_y):
        coordinates.append([int(np.round(float(x))), int(np.round(float(y)))])
    np.asarray(coordinates).astype(np.float32)
    return coordinates

def json2mask(mask_label_dict={}):
    def f(image=None, mask=None, mask_label_dict=mask_label_dict):
        h,w,c = image.shape
        mask_label_dict = { k.upper() : v for k,v in mask_label_dict.items() }
        segmentations = mask["polygonLabels"]
        mask = np.zeros([h,w], dtype=np.uint8)
        for cls_idx, segmentation in enumerate(segmentations):
            labelValue = segmentation['labelValue'].upper()
            allX, allY = [], []
            try:
                allX, allY = segmentation['all_points_x'], segmentation['all_points_y']
            except:
                try:
                    allX, allY = segmentation['allX'], segmentation['allY']
                except:
                    LOG.warning("no coordinates in segmentation entry: {}".format(segmentation))
            if len(allX) > 0 and len(allY) > 0 and len(allX) == len(allY):
                coordinates = get_coordinates(allX, allY)
                if labelValue not in mask_label_dict:
                    LOG.warning("found annotation label not in label dictionary: {}".format(labelValue))
                elif labelValue in mask_label_dict:
                    get_mask_from_coordinates(coordinates, [h,w], value=mask_label_dict[labelValue], mask=mask)
                elif mask_label_dict == {}:
                    get_mask_from_coordinates(coordinates, [h,w], value=cls_idx + 1, mask=mask)
        return image, mask
    return f

import torch

def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    hsl_h /= 6.

    hsl_l = (cmax + cmin) / 2.
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- torch.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c = (-torch.abs(hsl_l * 2. - 1.) + 1) * hsl_s
    _x = _c * (-torch.abs(hsl_h * 6. % 2. - 1) + 1.)
    _m = hsl_l - _c / 2.
    idx = (hsl_h * 6.).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb