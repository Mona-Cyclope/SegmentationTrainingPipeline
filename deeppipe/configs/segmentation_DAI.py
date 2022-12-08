# Configuration of the training
from deeppipe.datasets.augmentations import ApplyAlbumination
from deeppipe.datasets.datasets import ImageMaskDataset
from deeppipe.datasets.preprocess import image_mask_resize, json2mask, convertImage2float, convertImageRgb2Hsv, normCeneterImage, convertMask2long, transposeImage, compose
from deeppipe.datasets.io import load_image, load_json
import albumentations as A
from deeppipe.models.unet import UnetTrainer
from datetime import datetime

batches = { "batch_16_dirif_20191021": { 
                "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/DIRIF/batch_16_dirif_20191021/images",
                "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/DIRIF/batch_16_dirif_20191021/mask_labels",
            },
           "batch_17_dirif_comptage": {
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/DIRIF/batch_17_dirif_comptage/images",
                "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/DIRIF/batch_17_dirif_comptage/mask_labels",
           },
           "batch_18_dirif_AlarmesOctobre": {
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/DIRIF/batch_18_dirif_AlarmesOctobre/images",
                "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/DIRIF/batch_18_dirif_AlarmesOctobre/mask_labels",
           },
           "TOULON_DAI/batch_1":{
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_1/images",
               "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_1/mask_labels",
               
           },
           "TOULON_DAI/batch_2":{
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_2/images",
               "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_2/mask_labels",
               
           },
           "TOULON_DAI/batch_3":{
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_3/images",
               "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_3/mask_labels",
               
           },
           "TOULON_DAI/batch_4":{
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_4/images",
               "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_4/mask_labels",
               
           },
           "TOULON_DAI/batch_5":{
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_5/images",
               "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_5/mask_labels",
               
           },
           "TOULON_DAI/batch_6":{
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_6/images",
               "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_6/mask_labels",
               
           },
           "TOULON_DAI/batch_7":{
               "image_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_7/images",
               "mask_folder": "/raid-dgx1/allanza/Segmentation/semantic_segmentation/data/TOULON_DAI/batch_7/mask_labels",
           }
}

label_dict = {"Road":1, "Bau":2, "Sw":3}
resize_image_size = [400,400]
image_mask_preproc_fun = compose([ json2mask(label_dict), image_mask_resize(resize_image_size), convertMask2long,
                                  convertImageRgb2Hsv, convertImage2float, normCeneterImage(0,255),transposeImage])
image_load_fun = load_image
mask_load_fun = load_json
image_prefix = ""
mask_prefix = ""
image_suffix=".png"
mask_suffix=".json"

train_batch_size = 8
valid_batch_size = 8

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
channels_image = 3

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
fig_params = {"figsize": (12,12), "dpi":72, "sharex":True, "sharey":True, "alpha":0.5}

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

model = UnetTrainer(channels_image,n_classes)
model.set_figure_params(fig_params)
# convert to string
logging_folder = "/home/mviti/gits/SegmentationTrainingPipeline/log/DAI_DIRIF"
model_name = "Unet_{}".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

