import json
import numpy as np
from decimal import Decimal
from sphdet.bbox.deg2kent_single import deg2kent_single
import torch
import logging
import os
import hashlib
import h5py

# Define constants
INPUT_FILENAME = 'datasets/360INDOOR/annotations/instances_train2017.json'
OUTPUT_FILENAME_TEMPLATE = 'datasets/360INDOOR/annotations_small/instances_train2017_transformed_{}.json'
IMAGE_LIMIT = 20000
LIMIT = 2000
MIN_BBOX_SIZE = 8
MIN_KAPPA = 10
MAX_KAPPA = 100
BETA_RATIO_THRESHOLD = 0.4

def setup_logging(output_filename):
    log_filename = os.path.splitext(output_filename)[0] + '_log.txt'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    return log_filename

def log_preamble():
    logging.info("Filtering Parameters:")
    logging.info(f"IMAGE_LIMIT: {IMAGE_LIMIT}")
    logging.info(f"LIMIT: {LIMIT}")
    logging.info(f"MIN_BBOX_SIZE: {MIN_BBOX_SIZE}")
    logging.info(f"MIN_KAPPA: {MIN_KAPPA}")
    logging.info(f"MAX_KAPPA: {MAX_KAPPA}")
    logging.info(f"BETA_RATIO_THRESHOLD: {BETA_RATIO_THRESHOLD}")

def transform_bbox(bbox):
    tensor_result = deg2kent_single(bbox, 480, 960)
    return tensor_result.tolist()

def filter_small_bfovs(obj):
    return obj['bbox'][2] >= MIN_BBOX_SIZE and obj['bbox'][3] >= MIN_BBOX_SIZE

def filter_kappas(obj):
    return MIN_KAPPA <= obj['bbox'][3] <= MAX_KAPPA

def filter_betas(obj):
    return obj['bbox'][3] / (obj['bbox'][4] + 1e-6) < 1 / BETA_RATIO_THRESHOLD

def generate_filename_hash(category_ids):
    hash_object = hashlib.md5(str(category_ids).encode())
    return hash_object.hexdigest()[:8]

def main():
    # Load data
    with open(INPUT_FILENAME, 'r') as file:
        json_data = json.load(file)

    annotations = json_data['annotations']
    images = json_data['images']
    
    # Filter images and annotations
    image_ids = set(img['id'] for img in images[:IMAGE_LIMIT])
    filtered_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
    filtered_images = [img for img in images if img['id'] in image_ids]

    # Generate output filename
    category_ids = [obj.get('category_id', 'unknown') for obj in filtered_annotations[:LIMIT]]
    filename_hash = generate_filename_hash(category_ids)
    output_filename = OUTPUT_FILENAME_TEMPLATE.format(filename_hash)

    # Setup logging
    log_filename = setup_logging(output_filename)
    log_preamble()

    # Process annotations
    transformed_annotations = []
    original_boxes = []
    transformed_boxes = []

    for obj in filtered_annotations[:LIMIT]:
        # Store original data
        original_bbox = obj['bbox'].copy()
        original_annotation = obj.copy()

        # Convert decimal values
        obj['bbox'] = [float(value) if isinstance(value, Decimal) else value for value in obj['bbox']]
        obj['area'] = float(obj['area']) if isinstance(obj['area'], Decimal) else obj['area']

        # Transform bbox
        transformed_bbox = transform_bbox(torch.tensor(obj['bbox']))[0]
        obj['bbox'] = transformed_bbox
        
        # Apply filters
        if not filter_kappas(obj):
            logging.info(f"Filtered out due to kappa: {obj['bbox']}")
            continue
        if filter_betas(obj):
            logging.info(f"Filtered out due to beta ratio: {obj['bbox']}")
            continue
        if abs(obj['bbox'][2]) < 1e-6:
            obj['bbox'][2] = 0

        # Store boxes and annotations
        original_boxes.append(original_bbox)
        transformed_boxes.append(transformed_bbox)
        transformed_annotations.append(original_annotation)

        logging.info(f"Transformed bbox from {original_bbox} to {transformed_bbox}")

    # Convert to numpy arrays
    original_boxes = np.array(original_boxes)
    transformed_boxes = np.array(transformed_boxes)

    # Save boxes to HDF5
    with h5py.File("bbox_comparison_easy.h5", 'w') as f:
        f.create_dataset('original_boxes', data=original_boxes)
        f.create_dataset('transformed_boxes', data=transformed_boxes)

    # Save annotations to JSON
    json_data['annotations'] = transformed_annotations
    json_data['images'] = filtered_images
    with open(output_filename, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    print(f"Transformation log file created: {log_filename}")
    print(f"Saved {len(original_boxes)} original and transformed boxes to bbox_comparison.h5")

if __name__ == "__main__":
    main()