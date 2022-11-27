# SegmentationTrainingPipeline
Segmentation Training Pipeline with pytorch and augmentation

## Training pipeline CUDA 10/11

Two dedicated environment have been created these map to two different pytorch version
Stable (1.13) cuda 11 and LTS (1.8) cuda 10 python<=3.8

conda create -f train_py37_th_[8,13]_cu[10,11].yml


## ImageMaskDataloader with json from labeler

You can create a dataloader strating from a json labeler file. https://labeler.cyclope.ai/


