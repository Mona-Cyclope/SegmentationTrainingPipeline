
import numpy as np

def ApplyAlbumination(aug):
    def f(image, mask):
        augmented = aug(image=image, mask=mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        return image_aug, mask_aug
    return f

def compose(fun_list):
    def apply(image=None, mask=None):
        for f in fun_list:
            image, mask = f(image=image, mask=mask)
        return image, mask
    return apply

def convertImage2float(image=None, mask=None): return image.astype(np.float32), mask
def convertMask2long(image=None, mask=None): return image, mask.astype(np.int64)
def transposeImage(image=None, mask=None): return np.transpose(image, [2,0,1]), mask
