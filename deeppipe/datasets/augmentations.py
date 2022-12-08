
def ApplyAlbumination(aug):
    def f(image, mask):
        augmented = aug(image=image, mask=mask)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        return image_aug, mask_aug
    return f

