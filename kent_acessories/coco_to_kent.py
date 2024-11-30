import json
import torch
import numpy as np
import h5py
from sphdet.bbox.deg2kent_single import deg2kent_single
import pdb

def convert_coco_annotations(annotations_file, h=480, w=960):
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    original_annotations = []
    kent_params = []
    
    for ann in data['annotations']:
        bbox = ann['bbox']
        
        # Convert to spherical coordinates
        x, y, width, height = bbox
        eta = (x + width/2) * 360.0 / w
        alpha = (y + height/2) * 180.0 / h
        fov_h = width * 360.0 / w
        fov_v = height * 180.0 / h
        
        original_ann = [eta, alpha, fov_h, fov_v]
        
        # Convert to Kent parameters
        annotation = torch.tensor(original_ann, dtype=torch.float32)
        kent = deg2kent_single(annotation, h, w)
        kent = kent.squeeze(0)
        
        # Clamp beta to be non-negative
        kent[4] = max(kent[4], 0)
        
        # First check the tensor size
        if kent.size(0) < 5:  # Kent distribution should have 5 parameters
            print(f"Skipping due to incorrect parameter count: kent.size = {kent.size()}")
            continue
            
        # Then proceed with other checks
        if torch.isnan(kent).any():
            print(f"Skipping due to NaN: Original annotation = {original_ann}, Kent params = {kent}")
            continue
        if kent[3] <= 0:
            print(f"Skipping due to kappa <= 0: Original annotation = {original_ann}, kappa = {kent[3]}")
            continue
        if kent[4] < 0:
            print(f"Skipping due to beta <= 0: Original annotation = {original_ann}, beta = {kent[4]}")
            continue
        if kent[4] >= kent[3]/2:
            print(f"Skipping due to beta >= kappa/2: Original annotation = {original_ann}, kappa = {kent[3]}, beta = {kent[4]}")
            continue
            
        original_annotations.append(original_ann)
        kent_params.append(kent.detach().cpu().numpy())

    return np.array(original_annotations), np.array(kent_params)

if __name__ == "__main__":
    annotations_file = "datasets/360INDOOR/annotations/instances_val2017.json"
    output_file = "kent_samples.h5"
    
    original_annotations, kent_params = convert_coco_annotations(annotations_file)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('original_annotations', data=original_annotations)
        f.create_dataset('new_annotations', data=kent_params)
        
        print("\nSample Information:")
        print(f"Number of samples: {len(original_annotations)}")
        print(f"Original annotations shape: {original_annotations.shape}")
        print(f"Kent parameters shape: {kent_params.shape}")
        print(f"Saved samples to {output_file}")
        
        print("\nVerification:")
        print("Original annotations range:")
        print(f"X min: {np.min(original_annotations, axis=0)}")
        print(f"X max: {np.max(original_annotations, axis=0)}")
        print("Kent parameters range:")
        print(f"y min: {np.min(kent_params, axis=0)}")
        print(f"y max: {np.max(kent_params, axis=0)}")