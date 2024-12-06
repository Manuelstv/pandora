import json
import numpy as np

def analyze_coco_ranges(json_file):
    """
    Analyze the ranges of variables in COCO format annotations
    
    Parameters:
    json_file (str): Path to the COCO format JSON file
    
    Returns:
    dict: Statistics for each analyzed field
    """
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize dictionaries to store values
    stats = {
        'image_ids': set(),
        'category_ids': set(),
        'annotation_ids': set(),
        'bbox_x': [],
        'bbox_y': [],
        'bbox_width': [],
        'bbox_height': [],
        'areas': [],
        'iscrowd': set()
    }
    
    # Collect values from annotations
    for ann in data['annotations']:
        stats['annotation_ids'].add(ann['id'])
        stats['image_ids'].add(ann['image_id'])
        stats['category_ids'].add(ann['category_id'])
        
        # Extract bbox values
        bbox = ann['bbox']
        stats['bbox_x'].append(bbox[0])
        stats['bbox_y'].append(bbox[1])
        stats['bbox_width'].append(bbox[2])
        stats['bbox_height'].append(bbox[3])
        
        stats['areas'].append(ann['area'])
        stats['iscrowd'].add(ann['iscrowd'])
    
    # Calculate statistics
    results = {
        'Annotation IDs': {
            'min': min(stats['annotation_ids']),
            'max': max(stats['annotation_ids']),
            'unique_count': len(stats['annotation_ids'])
        },
        'Image IDs': {
            'min': min(stats['image_ids']),
            'max': max(stats['image_ids']),
            'unique_count': len(stats['image_ids'])
        },
        'Category IDs': {
            'min': min(stats['category_ids']),
            'max': max(stats['category_ids']),
            'unique_count': len(stats['category_ids']),
            'values': sorted(list(stats['category_ids']))
        },
        'Bounding Boxes': {
            'x': {
                'min': min(stats['bbox_x']),
                'max': max(stats['bbox_x']),
                'mean': np.mean(stats['bbox_x']),
                'std': np.std(stats['bbox_x'])
            },
            'y': {
                'min': min(stats['bbox_y']),
                'max': max(stats['bbox_y']),
                'mean': np.mean(stats['bbox_y']),
                'std': np.std(stats['bbox_y'])
            },
            'width': {
                'min': min(stats['bbox_width']),
                'max': max(stats['bbox_width']),
                'mean': np.mean(stats['bbox_width']),
                'std': np.std(stats['bbox_width'])
            },
            'height': {
                'min': min(stats['bbox_height']),
                'max': max(stats['bbox_height']),
                'mean': np.mean(stats['bbox_height']),
                'std': np.std(stats['bbox_height'])
            }
        },
        'Areas': {
            'min': min(stats['areas']),
            'max': max(stats['areas']),
            'mean': np.mean(stats['areas']),
            'std': np.std(stats['areas'])
        },
        'Is Crowd': {
            'values': sorted(list(stats['iscrowd']))
        }
    }
    
    return results

def print_analysis(results):
    """Pretty print the analysis results"""
    print("\n=== COCO Dataset Analysis ===\n")
    
    for category, values in results.items():
        print(f"\n{category}:")
        if isinstance(values, dict):
            if 'values' in values:
                print(f"  Unique values: {values['values']}")
            else:
                for metric, value in values.items():
                    print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
# Usage example
if __name__ == "__main__":
    # Replace with your JSON file path
    file_path = "datasets/360INDOOR/annotations/instances_val2017.json"
    results = analyze_coco_ranges(file_path)
    print_analysis(results)