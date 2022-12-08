import numpy as np
import json
import cv2
import os
import wget
from tqdm import tqdm
import pandas
import os
import json
from deeppipe import LOG

def load_json(path2file):
    with open(path2file, 'r', encoding='utf-8') as f:
        dictio = json.load(f)
    return dictio

def write_json(dictio, path2file):
    with open(path2file, "w") as f:
        json.dump(dictio, f)
    
def load_image(path_to_image):
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

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
    
def convert_csv_to_labeler(path, mask_folder_name='mask_labels'):
    mask_save_path = os.path.join(path, mask_folder_name)
    os.makedirs(mask_save_path, exist_ok=True)
    annotation_csv = [ pandas.read_csv(os.path.join(path,f)) for f in os.listdir(path) if '.csv' in f ]
    annotation_csv_fname = [ f for f in os.listdir(path) if '.csv' in f ]
    annotation_csv[0]
    filenames = list(set().union( *[ set(csv['#filename']) for csv in annotation_csv] ))
    annotations = {}
    for filename in filenames:
        annotations[filename] = {'polygonLabels':[]}
        for csv, csv_fname in zip(annotation_csv, annotation_csv_fname):
            entry = csv[csv[ "#filename"] == filename ]
            try:
                annotation_dict = json.loads(entry["region_shape_attributes"].values[0])
                annotation_class = json.loads(entry["region_attributes"].values[0])
                label = annotation_class['label']
                annotation_dict["labelValue"] = label[0].upper() + label[1:].lower()
                annotations[filename]['polygonLabels'] += [ annotation_dict ]
            except Exception as e:
                LOG.warning("issue with {} in {} raised: {}".format(filename, csv_fname, e))

    for k, v in annotations.items():
        fn = k.split('.png')[0]
        json_path = os.path.join(mask_save_path, "{}.json".format(fn))
        with open(json_path, 'w') as fp:
            json.dump(v, fp)