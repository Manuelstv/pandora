import json
import cv2
import numpy as np

def read_coco_json(image_name, annotations_file):
    """
    Extract all bounding boxes associated with a specific image name from COCO annotations
    
    Args:
        image_name (str): The filename of the image (e.g., "000684.jpg")
        annotations_file (str): Path to the COCO annotations JSON file
    
    Returns:
        list: List of dictionaries containing bounding box info
        Each dict contains: {
            'category_id': int,
            'bbox': [x, y, width, height],
            'area': float,
            'iscrowd': int
        }
    """
    boxes = []
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # First find the image ID for this filename
    image_id = None
    for img in data['images']:
        if img['file_name'] == image_name:
            image_id = img['id']
            break
            
    if image_id is None:
        raise ValueError(f"Image {image_name} not found in annotations")
        
    # Now filter annotations for this image ID
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            boxes.append({
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann['iscrowd']
            })
            
    return boxes

# Usage example
if __name__ == "__main__":
    # Configuration
    image_path = "datasets/360INDOOR/images/7fB4v.jpg"  # Adjust path as needed
    annotations_file = "datasets/360INDOOR/annotations/instances_val2017.json"
    
    # Get image name from path
    image_name = image_path.split('/')[-1]
    
    # Get boxes
    try:
        boxes =read_coco_json(image_name, annotations_file)
        print(f"Found {len(boxes)} boxes for image {image_name}")
        
        # Create list of bounding boxes
        bbox_list = [box['bbox'] for box in boxes]
        classes_list = [box['category_id'] for box in boxes]
        
        # Print box details
        '''print("\nBox details:")
        for i, box in enumerate(boxes, 1):
            print(f"\nBox {i}:")
            print(f"Category ID: {box['category_id']}")
            print(f"Coordinates [x,y,w,h]: {box['bbox']}")
            print(f"Area: {box['area']}")
            print(f"Is Crowd: {box['iscrowd']}")'''

        print(bbox_list)
        print(classes_list)
            
    except Exception as e:
        print(f"Error: {str(e)}")