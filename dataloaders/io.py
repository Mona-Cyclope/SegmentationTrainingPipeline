import numpy as np
import cv2

def get_mask_from_coordinates(coordinates, shape):
    mask = np.zeros(shape[0:2], dtype=np.uint8)
    hull = np.expand_dims(np.asarray(coordinates), axis=1)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    return mask


def get_coordinates(all_points_x, all_points_y):
    coordinates = []  # np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    for x, y in zip(all_points_x.split(";"), all_points_y.split(";")):
        coordinates.append([int(np.round(float(x))), int(np.round(float(y)))])

    np.asarray(coordinates).astype(np.float32)

    return coordinates

def get_lot_from_labeler(json_file_path, root_dump_path=None, update_if_exists=True):
    
    batch_name = os.path.basename(json_file_path).split('.')[0]
    root_dump_path = os.path.dirname(json_file_path) if root_dump_path is None else root_dump_path
    dump_path = os.path.join(root_dump_path, batch_name)
    if os.path.exists(dump_path) and not update_if_exists: return dump_path
    
    # open json file
    with open(json_file_path, "r") as f:
        batch_data = json.load(f)
    
    image_dump_path = os.path.join(dump_path, "images")
    labels_dump_path = os.path.join(dump_path, "labels")
    os.makedirs(image_dump_path, exist_ok=True)
    os.makedirs(labels_dump_path, exist_ok=True)
    
    for entry in batch_data:
        idx = entry['id']
        name = entry['name']
        url = entry['url']
        labels = entry['labels']
        wget.download(url, image_dump_path)
    
    return dump_path
    
    
    