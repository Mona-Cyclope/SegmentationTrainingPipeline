import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from deeppipe.datasets.io import image_load_fun, mask_load_fun, rescale_image_mask_load
from deeppipe import LOG
import cv2

def image_mask_load_fun(image_mask_path):
    image_path, mask_path = image_mask_path
    return image_load_fun(image_path), mask_load_fun(mask_path)

def get_name_of_file(path_to_file, prefix="", suffix=""):
    filename = os.path.basename(path_to_file)[len(prefix):-len(suffix)]
    return filename
    
def filter_file_names(folder, prefix, suffix):
    image_names = [ get_name_of_file(f, prefix=prefix, suffix=suffix ) 
                    for f in os.listdir(folder) if prefix in f and suffix in f ]
    return image_names

class BufferedDataloader(Dataset):
    
    def __init__(self, paths_to_files, read_file_fun):
        self.paths_to_files= paths_to_files
        self.read_file_fun = read_file_fun
        self.___buffer_data_loader = [ read_file_fun(path_to_file) for path_to_file in tqdm(paths_to_files, desc="BufferedDataloader") ]
        
    def __len__(self): return len(self.___buffer_data_loader)
    
    def __getitem__(self, idx):
        return self.___buffer_data_loader[idx]

class ImageMaskDataset(Dataset):
    
    def __init__(self, image_folder, mask_folder, image_prefix="", mask_prefix="", image_suffix=".png", mask_suffix=".png",
                 image_mask_load_fun=image_mask_load_fun, image_mask_transform=None):
        
        image_names = set(filter_file_names(image_folder, prefix=image_prefix, suffix=image_suffix))
        mask_names = set(filter_file_names(mask_folder, prefix=mask_prefix, suffix=mask_suffix))
        valid_file_names = list(image_names.intersection(mask_names))   

        valid_image_file_names = [ image_prefix+f+image_suffix for f in valid_file_names ]
        valid_mask_file_names = [ mask_prefix+f+mask_suffix for f in valid_file_names ]
        
        valid_image_file_paths = [ os.path.join(image_folder, f) for f in valid_image_file_names ]
        valid_mask_file_paths = [ os.path.join(mask_folder, f) for f in valid_mask_file_names ]
        
        valid_image_mask_paths = list(zip(valid_image_file_paths, valid_mask_file_paths))
        self.___buffer_data_loader = BufferedDataloader(valid_image_mask_paths, image_mask_load_fun)
        self.transform = image_mask_transform
        
    def __len__(self):
        return len(self.___buffer_data_loader)
    
    def __getitem__(self, index):
        image, mask = self.___buffer_data_loader[index]
        if self.transform:
            return self.transform(image=image, mask=mask)
        return image, mask
    
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
    