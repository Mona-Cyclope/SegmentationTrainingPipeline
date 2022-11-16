import os
from torch.utils.data import Dataset

def get_name_of_file(path_to_file, suffix="", ext=""):
    filename = os.path.basename(path_to_file)[len(suffix)::-len(ext)]
    return filename
    
def filter_file_names(folder, suffix, ext):
    image_names = [ get_name_of_file(f, suffix=suffix, ext=ext ) 
                    for f in os.listdir(folder) if suffix in f and ext in f ]
    return image_names
    
class BufferedDataloader(Dataset):
    
    def __init__(self, paths_to_files, read_file_fun):
        self.paths_to_files= paths_to_files
        self.read_file_fun = read_file_fun
        self.buffer = [ read_file_fun(path_to_file) for path_to_file in paths_to_files ]
        
    def __len__(self): return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
        
class SegmentationDataloader(Dataset):
    
    def __init__(self, image_folder, mask_folder, read_image_fun, read_mask_fun, 
                 image_ext, mask_ext, image_suffix, mask_suffix, image_mask_transform=None):
        # check naming consistency
        image_names = set(filter_file_names(image_folder, ext=image_ext, suffix=image_suffix))
        mask_names = set(filter_file_names(mask_folder, ext=mask_ext, suffix=mask_suffix))
        valid_file_names = list(image_names.intersection(mask_names))
        
        valid_image_file_names = [ image_suffix+f+image_ext for f in valid_file_names ]
        valid_mask_file_names = [ mask_suffix+f+mask_ext for f in valid_file_names ]
        
        valid_image_file_paths = [ os.path.join(image_folder, f) for f in valid_image_file_names ]
        valid_mask_file_paths = [ os.path.join(mask_folder, f) for f in valid_mask_file_names ]
        
        buffered_image_dataloader = BufferedDataloader(valid_image_file_paths, read_image_fun)
        buffered_mask_dataloader = BufferedDataloader(valid_mask_file_paths, read_mask_fun)
        
        self.buffer_image = buffered_image_dataloader
        self.buffer_mask = buffered_mask_dataloader
        
        assert len(self.buffer_image) == len(self.buffer_mask), "inconsinstent dataset"
        
        self.transform = image_mask_transform
        
    def __len__(self):
        return len(self.buffer_image)
        
    def __getitem__(self, idx):
        image, mask = self.buffer_image[idx], self.buffer_mask[idx]
        if self.transform: 
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
        
        
        
        
                
        