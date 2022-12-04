import albumentations as A
from deeppipe.models.unet import UnetTrainer
from datetime import datetime

zip_path = "data/kitti_v1.zip"
zip_ext = "data"

# trainer 
max_epochs = 100
devices = [0]
accelerator = "gpu"
log_every_n_steps = 5
check_val_every_n_epoch = 5
train_batch_size = 32
valid_batch_size = 12
n_classes = 2
channels_image = 3

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
fig_params = {"figsize": (12,12), "dpi":72, "sharex":True, "sharey":True, "alpha":0.5}

# data augmentation
albumentation = A.Compose([
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
logging_folder = "/home/mviti/gits/SegmentationTrainingPipeline/log/Kitti"
model_name = "Unet_{}".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))