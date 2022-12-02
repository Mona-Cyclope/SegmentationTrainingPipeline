import albumentations as A
from deeppipe.models.unet import UnetTrainer

train_batch_folder = "/home/mviti/gits/SegmentationTrainingPipeline/data"
train_batches_json = ["20210511_seg-batch-37-2021-05-11.json", "20210511_seg-batch-38-2021-05-11.json", "20210511_seg-batch-38-2021-05-11.json", 
                      "20210511_seg-batch-39-2021-05-11.json", "20210511_seg-batch-40-2021-05-11.json", "20210511_seg-batch-41-2021-05-11.json",
                      "20210511_seg-batch-42-2021-05-11.json", "20210511_seg-batch-43-2021-05-11.json", "20210511_seg-batch-44-2021-05-11.json",
                      "20210511_seg-batch-45-2021-05-11.json", "20210511_seg-batch-46-2021-05-11.json", "20210511_seg-batch-47-2021-05-11.json",
                      "20210511_seg-batch-48-2021-05-11.json", "20210511_seg-batch-49-2021-05-11.json", "20210511_seg-batch-50-2021-05-11.json",
                      "20210511_seg-batch-51-2021-05-11.json", "20210511_seg-batch-52-2021-05-11.json", "20210511_seg-batch-53-2021-05-11.json",
                      "20210511_seg-batch-54-2021-05-11.json", "20210511_seg-batch-55-2021-05-11.json", "20210511_seg-batch-56-2021-05-11.json",
                      "20210511_seg-batch-57-2021-05-11.json", "20210511_seg-batch-58-2021-05-11.json", "20210511_seg-batch-59-2021-05-11.json",
                      "20210511_seg-batch-60-2021-05-11.json", "20210511_seg-batch-61-2021-05-11.json", "20210511_seg-batch-62-2021-05-11.json",
                      "20210511_seg-batch-62-2021-05-11.json", "20210511_seg-batch-63-2021-05-11.json", "20210511_seg-batch-64-2021-05-11.json",
                      "20210511_seg-batch-65-2021-05-11.json", "20210511_seg-batch-66-2021-05-11.json", "20210511_seg-batch-67-2021-05-11.json",
                    ]
valid_batch_folder = "/home/mviti/gits/SegmentationTrainingPipeline/data"
valid_batches_json = ["20210511_seg-batch-68-2021-05-11.json", "20210511_seg-batch-69-2021-05-11.json", "20210511_seg-batch-70-2021-05-11.json", 
                      "20210511_seg-batch-71-2021-05-11.json", "20210511_seg-batch-72-2021-05-11.json", "20210511_seg-batch-73-2021-05-11.json",
                      "20210511_seg-batch-74-2021-05-11.json",
                    ]
label_dict = {"Road": 1, "BAU":2, "SW": 3}

# data download and buffer
download_images = False
resize_image_factor = 2
resize_height_image = 720//resize_image_factor
resize_width_image = 576//resize_image_factor
channels_image = 3

# sanity check for data label dictionary
n_classes = len(list(label_dict.keys())) + 1
assert n_classes == max(list(label_dict.values())) + 1

# trainer 
max_epochs = 100
devices = [0]
accelerator = "gpu"
log_every_n_steps = 10
train_batch_size = 32
valid_batch_size = 32

# data augmentation
albumentation = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        ], p=0.8),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),    
    A.RandomGamma(p=0.8)])

model = UnetTrainer(channels_image,n_classes)
