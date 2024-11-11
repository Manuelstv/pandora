import json
from decimal import Decimal
from sphdet.bbox.kent_formator import deg2kent_single
import pdb
import torch
import logging
import os

# Define the filename of the JSON file
INPUT_FILENAME = 'datasets/360INDOOR/annotations/instances_train2017.json'
OUTPUT_FILENAME_TEMPLATE = 'datasets/annotations_small/instances_train2017_transformed_5.json'

# Define filtering parameters
IMAGE_LIMIT = 100  # Change this value to the desired number of images
LIMIT = 10  # Change this value to the desired number of objects
MIN_BBOX_SIZE = 8
MIN_KAPPA = 10
MAX_KAPPA = 100
BETA_RATIO_THRESHOLD = 2.5

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
    logging.info(f"BETA_RATIO_THRESHOLD: {BETA_RATIO_THRESHOLD}")

def transform_bbox(bbox):
    tensor_result = deg2kent_single(bbox)
    return tensor_result.tolist() # Convert Tensor to list

def filter_small_bfovs(obj):
    # Filter out boxes where the third or fourth element is less than MIN_BBOX_SIZE
    return obj['bbox'][2] >= MIN_BBOX_SIZE and obj['bbox'][3] >= MIN_BBOX_SIZE

def filter_low_kappas(obj):
    return obj['bbox'][3] >= MIN_KAPPA and obj['bbox'][3] <= MAX_KAPPA

def filter_betas(obj):
    return obj['bbox'][3]/(obj['bbox'][4]+1e-6) < BETA_RATIO_THRESHOLD

def main():
    # Read the entire JSON file
    with open(INPUT_FILENAME, 'r') as file:
        json_data = json.load(file)

    annotations = json_data['annotations']
    images = json_data['images']
    transformed_annotations = []

    # Filter images to include only those with IDs up to IMAGE_LIMIT
    filtered_images = [img for img in images if img['id'] <= IMAGE_LIMIT]

    # Get the image IDs of the filtered images
    image_ids = [img['id'] for img in filtered_images]

    # Filter annotations to include only those with the selected image IDs
    filtered_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]

    # Extract object names for the filename
    object_names = [obj.get('name', 'unknown') for obj in filtered_annotations[:LIMIT]]
    object_names_str = '_'.join(object_names)

    # Create the output filename with object names
    output_filename = OUTPUT_FILENAME_TEMPLATE.format(object_names_str)

    # Set up logging with the output filename
    log_filename = setup_logging(output_filename)

    # Log the preamble with filtering parameters
    log_preamble()

    original_annotations = []
    new_annotations = []

    for obj in filtered_annotations[:]:
        original_bbox = obj['bbox'].copy()
        original_area = obj['area']

        # Convert Decimal to float if needed
        obj['bbox'] = [float(value) if isinstance(value, Decimal) else value for value in obj['bbox']]
        obj['area'] = float(obj['area']) if isinstance(obj['area'], Decimal) else obj['area']
        
        # Filter out small boxes
        if not filter_small_bfovs(obj):
            logging.info(f"Filtered out small bbox: {original_bbox}")
            continue
        
        # Transform the bbox values
        obj['bbox'] = transform_bbox(torch.tensor(obj['bbox']))[0]
        logging.info(f"Transformed bbox from {original_bbox} to {obj['bbox']}")
        
        if not filter_low_kappas(obj):
            logging.info(f"Filtered out due to low kappa: {obj['bbox']}")
            continue
        
        if filter_betas(obj):
            logging.info(f"Filtered out due to beta ratio: {obj['bbox']}")
            continue
        
        if abs(obj['bbox'][2]) < 1e-6:
            obj['bbox'][2] = 0
        
        # Print original and new annotations
        print(f"Original annotation: {original_bbox}, New annotation: {obj['bbox']}")
        
        # Add the original and transformed object to the lists
        original_annotations.append(original_bbox)
        new_annotations.append(obj['bbox'])

        # Add the transformed object to the list
        transformed_annotations.append(obj)

    # Update the annotations and images in the original JSON structure
    json_data['annotations'] = transformed_annotations
    json_data['images'] = filtered_images

    # Write the updated JSON structure to a new file
    with open(output_filename, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

    # Print the log file location
    #print(f"Transformation log file created: {log_filename}")

    return original_annotations, new_annotations

if __name__ == "__main__":
    original_annotations, new_annotations = main()
    #print("Original Annotations:", original_annotations)
    #print("New Annotations:", new_annotations)