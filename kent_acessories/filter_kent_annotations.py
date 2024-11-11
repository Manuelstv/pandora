import json
from decimal import Decimal
from sphdet.bbox.deg2kent_single import deg2kent_single
import torch
import logging
import os
import hashlib

# Define the filename of the JSON file
INPUT_FILENAME = 'datasets/360INDOOR/annotations/instances_train2017.json'
OUTPUT_FILENAME_TEMPLATE = 'datasets/360INDOOR/annotations_small/instances_train2017_transformed_{}.json'

# Define filtering parameters
IMAGE_LIMIT = 100  # Change this value to the desired number of images
LIMIT = 100  # Change this value to the desired number of objects
MIN_BBOX_SIZE = 8
MIN_KAPPA = 10
MAX_KAPPA = 50
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
    tensor_result = deg2kent_single(bbox)
    return tensor_result.tolist()  # Convert Tensor to list

def filter_small_bfovs(obj):
    # Filter out boxes where the third or fourth element is less than MIN_BBOX_SIZE
    return obj['bbox'][2] >= MIN_BBOX_SIZE and obj['bbox'][3] >= MIN_BBOX_SIZE

def filter_kappas(obj):
    #import pdb; pdb.set_trace()
    return MIN_KAPPA <= obj['bbox'][3] <= MAX_KAPPA

def filter_betas(obj):
    return obj['bbox'][3] / (obj['bbox'][4] + 1e-6) < 1 / BETA_RATIO_THRESHOLD

def generate_filename_hash(category_ids):
    # Generate a hash of the category IDs
    hash_object = hashlib.md5(str(category_ids).encode())
    return hash_object.hexdigest()[:8]  # Use first 8 characters of the hash

def main():
    # Read the entire JSON file
    with open(INPUT_FILENAME, 'r') as file:
        json_data = json.load(file)

    annotations = json_data['annotations']
    images = json_data['images']
    transformed_annotations = []

    image_ids = set(img['id'] for img in images[:IMAGE_LIMIT])
    filtered_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
    filtered_images = [img for img in images if img['id'] in image_ids]

    category_ids = [obj.get('category_id', 'unknown') for obj in filtered_annotations[:LIMIT]]
    filename_hash = generate_filename_hash(category_ids)

    output_filename = OUTPUT_FILENAME_TEMPLATE.format(filename_hash)

    log_filename = setup_logging(output_filename)

    log_preamble()

    for obj in filtered_annotations[:LIMIT]:
        original_bbox = obj['bbox'].copy()
        original_area = obj['area']

        original_annotation = obj.copy()

        obj['bbox'] = [float(value) if isinstance(value, Decimal) else value for value in obj['bbox']]
        obj['area'] = float(obj['area']) if isinstance(obj['area'], Decimal) else obj['area']

        obj['bbox'] = transform_bbox(torch.tensor(obj['bbox']))[0]
        logging.info(f"Transformed bbox from {original_bbox} to {obj['bbox']}")

        if not filter_kappas(obj):
            logging.info(f"Filtered out due to kappa: {obj['bbox']}")
            continue

        if filter_betas(obj):
            logging.info(f"Filtered out due to beta ratio: {obj['bbox']}")
            continue

        if abs(obj['bbox'][2]) < 1e-6:
            obj['bbox'][2] = 0

        transformed_annotations.append(original_annotation)

    json_data['annotations'] = transformed_annotations
    json_data['images'] = filtered_images

    with open(output_filename, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    print(f"Transformation log file created: {log_filename}")

if __name__ == "__main__":
    main()