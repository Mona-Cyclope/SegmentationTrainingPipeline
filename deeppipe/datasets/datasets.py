import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from deeppipe import LOG
import cv2

class ImageMaskPathsDataset(Dataset):
    
    def __init__(self, valid_image_file_paths, valid_mask_file_paths, image_load_fun, mask_load_fun,
                 image_mask_transform=None, image_mask_preproc_fun=None ):
        
        self.buffer_data = [ (image_load_fun(image_path), mask_load_fun(mask_path)) for (image_path, mask_path) in tqdm(zip(valid_image_file_paths, valid_mask_file_paths), desc='load in buffer') ]
        if image_mask_preproc_fun is not None:
            for idx in tqdm(range(len(self.buffer_data)), desc='preprocessing'):
                image, mask = self.buffer_data[idx]
                self.buffer_data[idx] = image_mask_preproc_fun(image=image, mask=mask)

        self.transform = image_mask_transform
        
    def __len__(self):
        return len(self.buffer_data)
    
    def __getitem__(self, index):
        image, mask = self.buffer_data[index]
        if self.transform: image, mask = self.transform(image=image, mask=mask)
        return image, mask
    
class ImageMaskDataset(ImageMaskPathsDataset):
    
    @staticmethod
    def filter_file_names(folder, prefix, anyfix, suffix):
        image_names = [ os.path.basename(f)[len(prefix):-len(suffix)] for f in os.listdir(folder) if prefix in f and suffix in f and anyfix in f ]
        return image_names
    
    def __init__(self, image_folder, mask_folder, image_load_fun, mask_load_fun,
                 image_mask_transform=None, image_mask_preproc_fun=None, 
                 image_prefix="", frame_code="", mask_prefix="", image_suffix=".png", mask_suffix=".json"
                 ):
        """
        prefix,frame_code,suffix: 
            ..../mask/mask_000_1.json, .../image/image_000_1.png -> image_name = _1, frame_code=000, image_prefix='image', mask_prefix='mask' image_suffix='.png' ...
        
        """
        if not isinstance(image_folder, list): image_folder = [image_folder]
        if not isinstance(mask_folder, list): mask_folder = [mask_folder]
            
        image_folders = image_folder
        mask_folders = mask_folder
        
        valid_image_file_paths = []
        valid_mask_file_paths = []
        
        for image_folder, mask_folder in zip(image_folders, mask_folders):
            image_names = set(ImageMaskDataset.filter_file_names(image_folder, prefix=image_prefix, anyfix=frame_code, suffix=image_suffix))
            mask_names = set(ImageMaskDataset.filter_file_names(mask_folder, prefix=mask_prefix, anyfix=frame_code, suffix=mask_suffix))
            valid_file_names = list(image_names.intersection(mask_names))   

            valid_image_file_names = [ image_prefix+f+image_suffix for f in valid_file_names ]
            valid_mask_file_names = [ mask_prefix+f+mask_suffix for f in valid_file_names ]
            
            valid_image_file_paths += [ os.path.join(image_folder, f) for f in valid_image_file_names ]
            valid_mask_file_paths += [ os.path.join(mask_folder, f) for f in valid_mask_file_names ]
        
        super().__init__(valid_image_file_paths, valid_mask_file_paths, image_load_fun, mask_load_fun,
                         image_mask_transform=image_mask_transform, image_mask_preproc_fun=image_mask_preproc_fun)
    
class KittiDataset(ImageMaskDataset):
    
    @staticmethod
    def unzip_kitti(zip_path, zip_ext):
        ex_dir = os.path.join(zip_ext, 'kitti')
        os.makedirs(ex_dir, exist_ok=True)
        os.system("unzip {} -d {} -n".format(zip_path, ex_dir))
        gt_path = os.path.join(ex_dir, "training", "gt_image_2")
        for fname in os.listdir(gt_path):
            try:
                code,clss,idx = fname.split('_')
                idx = idx.split('.png')[0]
                if clss == 'road':
                    refname = fname.replace('_road_','_')
                    os.rename(os.path.join(gt_path, fname), os.path.join(gt_path, refname))
                    
            except Exception as e:
                LOG.warning(e)
                LOG.warning("issue with {}, it might be already be renamed...".format(fname))
        return ex_dir
    
    @staticmethod
    def kitti_image_mask_load_fun(image_mask_path):
        image_path, mask_path = image_mask_path
        image,mask = image_load_fun(image_path), image_load_fun(mask_path)
        dim = 1200,350
        image = cv2.resize(image, tuple(dim), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, tuple(dim), interpolation=cv2.INTER_NEAREST)
        # convert to binary mask
        bmask = (mask.mean(axis=-1)>100).astype(np.uint8)
        return image, bmask
    
    def __init__(self, image_folder=None, mask_folder=None, zip_path=None, zip_ext=None, **dataset_kwargs):
        if image_folder is None and mask_folder is None:
            assert zip_path is not None and zip_ext is not None, "provide path to zip file"
            ex_dir = KittiDataset.unzip_kitti(zip_path, zip_ext)
            image_folder = os.path.join(ex_dir, "training", "image_2")
            mask_folder = os.path.join(ex_dir, "training", "gt_image_2")
            self.video_test_folder = os.path.join(ex_dir, "testing")
        self.image_folder, self.mask_folder = image_folder, mask_folder
        super().__init__(image_folder=image_folder, mask_folder=mask_folder, 
                         image_mask_load_fun=KittiDataset.kitti_image_mask_load_fun, **dataset_kwargs)
    