from sphdet.bbox.deg2kent_single import deg2kent_single
import numpy as np
import torch
from itertools import product
import tqdm
import h5py

def generate_uniform_samples(n_samples_per_dim={'x': 36, 'y': 36, 'width': 36, 'height': 24},
                           ranges={'x': [0, 360], 
                                 'y': [0, 180],
                                 'width': [1, 60],
                                 'height': [1, 60]},
                           image_size=(480, 960)):
    
    # Generate uniform samples for each dimension
    x_samples = np.linspace(ranges['x'][0], ranges['x'][1], n_samples_per_dim['x'])
    y_samples = np.linspace(ranges['y'][0], ranges['y'][1], n_samples_per_dim['y'])
    w_samples = np.linspace(ranges['width'][0], ranges['width'][1], n_samples_per_dim['width'])
    h_samples = np.linspace(ranges['height'][0], ranges['height'][1], n_samples_per_dim['height'])
    
    # Calculate total number of combinations
    total_samples = (n_samples_per_dim['x'] * n_samples_per_dim['y'] * 
                    n_samples_per_dim['width'] * n_samples_per_dim['height'])
    
    print(f"Generating {total_samples} samples...")
    
    # Initialize arrays to store results
    original_annotations = []
    new_annotations = []
    
    # Generate all combinations with progress bar
    combinations = list(product(x_samples, y_samples, w_samples, h_samples))
    
    # Process in batches to avoid memory issues
    batch_size = 100
    n_batches = (len(combinations) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm.tqdm(range(n_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(combinations))
        batch_combinations = combinations[start_idx:end_idx]
        
        # Convert batch to tensor
        batch_tensor = torch.tensor(batch_combinations, dtype=torch.float32, requires_grad=True)
        
        # Get Kent parameters for batch
        kent_params = deg2kent_single(batch_tensor, *image_size)
        
        # Store results
        original_annotations.extend(batch_combinations)
        new_annotations.extend(kent_params.detach().cpu().numpy())
        
        # Optional: Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    X = np.array(original_annotations)
    y = np.array(new_annotations)
    
    return X, y

def save_samples(X, y, filename='kent_samples.h5'):
    """Save samples to HDF5 format"""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('original_annotations', data=X)
        f.create_dataset('new_annotations', data=y)
    print(f"Saved samples to {filename}")

def load_samples(filename='kent_samples.h5'):
    """Load samples from HDF5 format"""
    with h5py.File(filename, 'r') as f:
        X = f['original_annotations'][:]
        y = f['new_annotations'][:]
    return X, y

if __name__ == '__main__':
    # Generate samples
    X, y = generate_uniform_samples()
    
    # Print sample information
    print("\nSample Information:")
    print(f"Number of samples: {len(X)}")
    print(f"Original annotations shape: {X.shape}")
    print(f"Kent parameters shape: {y.shape}")
    
    # Save samples
    save_samples(X, y)
    
    # Example loading
    X_loaded, y_loaded = load_samples()
    
    # Verify data
    print("\nVerification:")
    print("Original annotations range:")
    print(f"X min: {X.min(axis=0)}")
    print(f"X max: {X.max(axis=0)}")
    print("\nKent parameters range:")
    print(f"y min: {y.min(axis=0)}")
    print(f"y max: {y.max(axis=0)}")