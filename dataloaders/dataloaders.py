import os
from torch.utils.data import Dataset
from .io import image_load_fun, mask_load_fun

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
        self.___buffer_data_loader = [ read_file_fun(path_to_file) for path_to_file in paths_to_files ]
        
    def __len__(self): return len(self.___buffer_data_loader)
    
    def __getitem__(self, idx):
        return self.___buffer_data_loader[idx]

class ImageMaskDataloader(Dataset):
    
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
        if self.transform:
            return self.transform(self.___buffer_data_loader[index])
        return self.___buffer_data_loader[index]