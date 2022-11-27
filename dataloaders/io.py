import numpy as np
import json
import cv2
import os
import wget
from tqdm import tqdm

from dataloaders import LOG

def load_json(path2file):
    with open(path2file, 'r', encoding='utf-8') as f:
        dictio = json.load(f)
    return dictio

def write_json(dictio, path2file):
    with open(path2file, "w") as f:
        json.dump(dictio, f)
    
def image_load_fun(path_to_image):
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def mask_load_fun(path_to_mask):
    with open(path_to_mask, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    segmentation = labels["polygonLabels"]
    return segmentation

def get_mask_from_coordinates(coordinates, shape, value=1, mask=None):
    H,W = shape
    mask = np.zeros([H,W], dtype=np.uint8) if mask is None else mask
    hull = np.expand_dims(np.asarray(coordinates), axis=1)
    cv2.drawContours(mask, [hull], -1, value, -1)
    return mask

def get_coordinates(all_points_x, all_points_y):
    coordinates = []
    for x, y in zip(all_points_x, all_points_y):
        coordinates.append([int(np.round(float(x))), int(np.round(float(y)))])
    np.asarray(coordinates).astype(np.float32)
    return coordinates

def labeler_image_mask_load_fun(image_mask_path):
    image_path, mask_path = image_mask_path
    image = image_load_fun(image_path)
    segmentations = mask_load_fun(mask_path)
    h,w,c = image.shape
    mask = np.zeros([h,w], dtype=np.uint8)
    for cls_idx,segmentation in enumerate(segmentations):
        cls = cls_idx + 1
        allX, allY = segmentation['allX'], segmentation['allY']
        coordinates = get_coordinates(allX, allY)
        get_mask_from_coordinates(coordinates, [h,w], value=cls, mask=mask)
    return image, mask

def labeler_download_lots(json_file_paths, update_if_exists=True, only_labels=False):
    dump_paths = [ labeler_download_lot(json_file_path, update_if_exists=update_if_exists, only_labels=only_labels) for json_file_path in json_file_paths ]
    image_paths, label_paths = list(zip(*dump_paths))
    return image_paths, label_paths

def labeler_download_lot(json_file_path, root_dump_path=None, update_if_exists=True, only_labels=False):
    """
    download data lot from json file in labeler format.
    images are downloaded as png
    labels in json format
    Returns:
        string: path to downloaded dump <lot name>/images <lot name>/labels
    """
    batch_name = os.path.basename(json_file_path).split('.')[0]
    root_dump_path = os.path.dirname(json_file_path) if root_dump_path is None else root_dump_path
    dump_path = os.path.join(root_dump_path, batch_name)
    if os.path.exists(dump_path) and not update_if_exists: return dump_path
    
    batch_data = load_json(json_file_path)
        
    image_dump_path = os.path.join(dump_path, "images")
    labels_dump_path = os.path.join(dump_path, "labels")
    os.makedirs(image_dump_path, exist_ok=True)
    os.makedirs(labels_dump_path, exist_ok=True)
    
    for entry in tqdm(batch_data, desc ="download_lot"):
        idx = entry['id']
        name = entry['name'][:-4]
        url = entry['url']
        labels = entry['labels']
        if not only_labels:
            try:
                wget.download(url, image_dump_path)
            except Exception as e:
                LOG.error(e)
        labels_dump_path_json = os.path.join(labels_dump_path, "{}.json".format(name))
        write_json(labels, labels_dump_path_json)
    
    return image_dump_path, labels_dump_path
    
    
    