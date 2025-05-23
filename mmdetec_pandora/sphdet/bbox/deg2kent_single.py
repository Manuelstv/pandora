import torch
import sys
import numpy as np
import torch.nn as nn
import pdb
from sphdet.bbox.kent_formator import kent_me_matrix_torch, get_me_matrix_torch
from memory_profiler import profile

def bfov_to_kent(annotations, epsilon=1e-6):
    """
    Converts bounding field of view (BFOV) annotations to Kent distribution parameters,
    ensuring stability and avoiding inf/nan values.

    Args:
        annotations (torch.Tensor): [n, 4] tensor where each row is [x, y, fov_h, fov_v].
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        torch.Tensor: [n, 5] tensor where each row is [eta, alpha, psi, kappa, beta].
    """
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)  # Convert to batch of size 1
    
    # Extract and normalize coordinates
    data_x = annotations[:, 0] / 360.0  # Normalize x coordinate (longitude)
    data_y = annotations[:, 1] / 180.0  # Normalize y coordinate (latitude)
    #data_z = annotations[:, 2] / 180.0 
    
    data_fov_h = annotations[:, 2]
    data_fov_v = annotations[:, 3]
        
    phi = 2 * np.pi * data_x            # Azimuthal angle (longitude)
    theta = np.pi * data_y              # Polar angle (latitude)
    psi = annotations[:, 4]
    
    varphi = (torch.deg2rad(data_fov_h) ** 2) / 12 + epsilon  # Horizontal variance
    vartheta = (torch.deg2rad(data_fov_v) ** 2) / 12 + epsilon  # Vertical variance
    
    kappa = 0.5 * (1 / varphi + 1 / vartheta)    
    beta = torch.abs(0.25 * (1 / vartheta - 1 / varphi))

    max_kappa = 1e3
    kappa = torch.clamp(kappa, max=max_kappa)
    beta = torch.clamp(beta, max=(kappa / 2) - 1e-4)
    
    # Set angles for Kent distribution
    eta = phi  # Azimuthal mean direction
    alpha = theta  # Polar mean direction
    
    # Stack parameters into [n, 5] tensor
    kent_dist = torch.stack([eta, alpha, psi, kappa, beta], dim=1)
    
    return kent_dist
    


if __name__ == '__main__':
    annotations = torch.tensor([35.0, 0.0, 23.0, 20.0], dtype=torch.float32, requires_grad=True)
    
    annotations_2 = torch.tensor([[180.28125, 133.03125,  5.     ,  5.    ], 
                            [35.0, 0.0, 23.0, 50.0], 
                            [35.0, 10.0, 23.0, 20.0]], dtype=torch.float32, requires_grad=True)
    print(annotations_2)

    kent = deg2kent_single(annotations_2, 480, 960)
    print("Kent:", kent)
    
    if not kent.requires_grad:
        kent = kent.detach().requires_grad_(True)
    
    loss = kent.sum()
    print("Loss:", loss)
    
    if loss.requires_grad:
        loss.retain_grad()
        loss.backward()
    else:
        print("Loss does not require gradients")
    
    print("Loss Grad:", loss.grad)
    print("Annotations Grad:", annotations_2.grad)