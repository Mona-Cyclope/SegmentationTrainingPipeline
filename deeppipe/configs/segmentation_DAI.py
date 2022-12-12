# Configuration of the training
from deeppipe.datasets.augmentations import ApplyAlbumination
from deeppipe.datasets.datasets import ImageMaskDataset
from deeppipe.datasets.preprocess import image_mask_resize, json2mask, convertImage2float, convertImageRgb2Hsv, normCeneterImage, convertMask2long, transposeImage, compose
from deeppipe.datasets.io import load_image, load_json
import albumentations as A
from deeppipe.models.unet import UnetHSVTrainer, UnetTrainer, UnetGRYTrainer
from deeppipe.models.denseaspp import DenseASPPGRYTrainer
from datetime import datetime
import pandas as pd

batches = {'20210511_seg-batch-37-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-37-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-37-2021-05-11/labels'},
 '20210511_seg-batch-38-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-38-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-38-2021-05-11/labels'},
 '20210511_seg-batch-39-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-39-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-39-2021-05-11/labels'},
 '20210511_seg-batch-40-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-40-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-40-2021-05-11/labels'},
 '20210511_seg-batch-41-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-41-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-41-2021-05-11/labels'},
 '20210511_seg-batch-42-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-42-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-42-2021-05-11/labels'},
 '20210511_seg-batch-43-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-43-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-43-2021-05-11/labels'},
 '20210511_seg-batch-44-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-44-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-44-2021-05-11/labels'},
 '20210511_seg-batch-45-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-45-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-45-2021-05-11/labels'},
 '20210511_seg-batch-46-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-46-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-46-2021-05-11/labels'},
 '20210511_seg-batch-47-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-47-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-47-2021-05-11/labels'},
 '20210511_seg-batch-48-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-48-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-48-2021-05-11/labels'},
 '20210511_seg-batch-49-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-49-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-49-2021-05-11/labels'},
 '20210511_seg-batch-50-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-50-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-50-2021-05-11/labels'},
 '20210511_seg-batch-51-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-51-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-51-2021-05-11/labels'},
 '20210511_seg-batch-52-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-52-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-52-2021-05-11/labels'},
 '20210511_seg-batch-53-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-53-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-53-2021-05-11/labels'},
 '20210511_seg-batch-54-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-54-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-54-2021-05-11/labels'},
 '20210511_seg-batch-55-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-55-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-55-2021-05-11/labels'},
 '20210511_seg-batch-56-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-56-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-56-2021-05-11/labels'},
 '20210511_seg-batch-57-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-57-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-57-2021-05-11/labels'},
 '20210511_seg-batch-58-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-58-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-58-2021-05-11/labels'},
 '20210511_seg-batch-59-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-59-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-59-2021-05-11/labels'},
 '20210511_seg-batch-60-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-60-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-60-2021-05-11/labels'},
 '20210511_seg-batch-61-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-61-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-61-2021-05-11/labels'},
 '20210511_seg-batch-62-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-62-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-62-2021-05-11/labels'},
 '20210511_seg-batch-63-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-63-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-63-2021-05-11/labels'},
 '20210511_seg-batch-64-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-64-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-64-2021-05-11/labels'},
 '20210511_seg-batch-65-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-65-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-65-2021-05-11/labels'},
 '20210511_seg-batch-66-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-66-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-66-2021-05-11/labels'},
 '20210511_seg-batch-68-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-68-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-68-2021-05-11/labels'},
 '20210511_seg-batch-69-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-69-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-69-2021-05-11/labels'},
 '20210511_seg-batch-70-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-70-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-70-2021-05-11/labels'},
 '20210511_seg-batch-71-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-71-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-71-2021-05-11/labels'},
 '20210511_seg-batch-72-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-72-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-72-2021-05-11/labels'},
 '20210511_seg-batch-73-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-73-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-73-2021-05-11/labels'},
 '20210511_seg-batch-74-2021-05-11': {'image_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-74-2021-05-11/images',
                                      'mask_folder': '/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/20210511_seg-batch-74-2021-05-11/labels'}
 }

camera_codes_paths = [ "/raid-dgx3/mviti/gits/SegmentationTrainingPipeline/data/Toulon/camera_codes.csv" ]

label_dict = {"Road":1, "Bau":2, "Sw":3, "Parking": 4}
train_valid_split = [0.9,0.1]
split_strategy = "by_camera_code" 
resize_image_size = [400,400]
image_mask_preproc_fun = compose([ json2mask(label_dict), image_mask_resize(resize_image_size), convertMask2long,
                                   convertImage2float, normCeneterImage(0,255),transposeImage])
image_load_fun = load_image
mask_load_fun = load_json
image_prefix = ""
mask_prefix = ""
image_suffix=".png"
mask_suffix=".json"

train_batch_size = 8
valid_batch_size = 8
precision = 32

logging_folder = "log"

# trainer 
max_epochs = 100
devices = [0]
accelerator = "gpu"
log_every_n_steps = 5
check_val_every_n_epoch = 5
train_batch_size = 16
valid_batch_size = 16
# sanity check for data label dictionary
n_classes = len(list(label_dict.keys())) + 1
assert n_classes == max(list(label_dict.values())) + 1
channels_image = 1

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
fig_params = {"figsize": (12,12), "dpi":90, "sharex":True, "sharey":True, "alpha":0.5}

# data augmentation
train_albumentation = A.Compose([
    A.RandomCrop(256,256,p=1.0),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.1),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
        A.GridDistortion(p=1.0),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0)                  
        ], p=0.5 ),
    A.RandomBrightnessContrast(p=0.25),    
    A.RandomGamma(p=0.25),
    A.Solarize (threshold=128, always_apply=False, p=0.25),
    A.OneOf([
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=1.0),
        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), 
                     blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=1.0),
        A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=1.0),
        A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=1.0),
        A.RandomSunFlare (flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, 
                         num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=1.0) 
    ], p=0.1)    
    ])


model = DenseASPPGRYTrainer(channels_image,n_classes, config='ASPP121')
model.set_figure_params(fig_params)
logging_folder = "/home/mviti/gits/SegmentationTrainingPipeline/log/DAI_TOULON"
model_name = "DenseASPPGRY_{}".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
#model = UnetGRYTrainer(channels_image,n_classes)
#model.set_figure_params(fig_params)
#logging_folder = "/home/mviti/gits/SegmentationTrainingPipeline/log/DAI_TOULON"
#model_name = "UnetGRY_{}".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))