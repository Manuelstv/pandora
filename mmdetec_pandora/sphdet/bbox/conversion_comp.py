import torch
import sys
import numpy as np
import torch.nn as nn
import pdb
from sphdet.bbox.kent_formator import kent_me_matrix_torch, get_me_matrix_torch
from sphdet.bbox.deg2kent_single import bfov_to_kent
from memory_profiler import profile

def project_equirectangular_to_sphere(u, w, h):
    """
    Projects equirectangular coordinates to spherical coordinates.

    Args:
        u (torch.Tensor): Input tensor with equirectangular coordinates.
        w (int): Width of the equirectangular image.
        h (int): Height of the equirectangular image.

    Returns:
        torch.Tensor: Tensor with spherical coordinates.
    """
    alpha = u[:, 1] * (torch.pi / float(h))
    eta = u[:, 0] * (2. * torch.pi / float(w))
    return torch.vstack([torch.cos(alpha), torch.sin(alpha) * torch.cos(eta), torch.sin(alpha) * torch.sin(eta)]).T

def sample_from_annotation_deg(annotation, shape):
    """
    Samples points from an annotation in degrees.

    Args:
        annotation (torch.Tensor): Tensor with annotation data.
        shape (tuple): Shape of the image (height, width).

    Returns:
        torch.Tensor: Sampled points in spherical coordinates.
    """
    h, w = shape
    device = annotation.device
    eta_deg, alpha_deg, fov_h, fov_v = annotation

    eta00 = torch.deg2rad(eta_deg - 180)
    alpha00 = torch.deg2rad(alpha_deg - 90)

    a_lat = torch.deg2rad(fov_v)
    a_long = torch.deg2rad(fov_h)

    r = 6
    epsilon = 1e-6  # Increased epsilon for better numerical stability
    d_lat = r / (2 * torch.tan(a_lat / 2 + epsilon))
    d_long = r / (2 * torch.tan(a_long / 2 + epsilon))

    i, j = torch.meshgrid(torch.arange(-(r - 1) // 2, (r + 1) // 2, device=device), 
                          torch.arange(-(r - 1) // 2, (r + 1) // 2, device=device), indexing='ij')
    
    p = torch.stack([i * d_lat / d_long, j, d_lat.expand_as(i)], dim=-1).reshape(-1, 3)

    sin_eta00, cos_eta00 = torch.sin(eta00), torch.cos(eta00)
    sin_alpha00, cos_alpha00 = torch.sin(alpha00), torch.cos(alpha00)

    Ry = torch.stack([
            cos_eta00, torch.zeros_like(eta00), sin_eta00,
            torch.zeros_like(eta00), torch.ones_like(eta00), torch.zeros_like(eta00),
            -sin_eta00, torch.zeros_like(eta00), cos_eta00
        ]).reshape(3, 3)
    
    Rx = torch.stack([
            torch.ones_like(alpha00), torch.zeros_like(alpha00), torch.zeros_like(alpha00),
            torch.zeros_like(alpha00), cos_alpha00, -sin_alpha00,
            torch.zeros_like(alpha00), sin_alpha00, cos_alpha00
        ]).reshape(3, 3)

    R = torch.matmul(Ry, Rx)

    if torch.isnan(R).any():
        raise ValueError("NaNs detected in rotation matrix R")

    p = torch.matmul(p, R.T)

    if torch.isnan(p).any():
        raise ValueError("NaNs detected in p after matrix multiplication")

    norms = torch.norm(p, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=epsilon)
    p = p / norms

    eta = torch.atan2(p[:, 0], p[:, 2])
    alpha = torch.clamp(p[:, 1], -1 + epsilon, 1 - epsilon)
    alpha = torch.asin(alpha)

    u = (eta / (2 * torch.pi) + 1. / 2.) * w
    v = h - (-alpha / torch.pi + 1. / 2.) * h
    return project_equirectangular_to_sphere(torch.vstack((u, v)).T, w, h)

from mlp_kent import *

#@profile
def deg2kent_single(annotations):
    """
    Converts annotations in degrees to Kent distribution parameters.
    Processes entire batch at once for better efficiency.

    Args:
        annotations (torch.Tensor): Tensor with annotation data [B, 4].
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        torch.Tensor: Kent distribution parameters [B, 5].
    """
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)  # Convert to batch of size 1

    model_path = 'kent_mlp_model.pth'
    model, scaler = load_kent_model(model_path)

    # Convert to radians in batch
    eta = (annotations[:, 0] / 360.0) * (2 * torch.pi)
    alpha = (annotations[:, 1] / 180.0) * torch.pi
    psi = torch.zeros_like(eta)  # Create zeros with same device and dtype

    # Get kappa predictions for entire batch
    kappa_predictions = predict_kappa(model, scaler, annotations)
    kappa, beta = kappa_predictions[:, 0], kappa_predictions[:, 1]

    # Stack all parameters into final tensor [B, 5]
    kent_params = torch.stack([eta, alpha, psi, kappa, beta], dim=1)
    
    return kent_params

def deg2kent_sampling(annotations, h=128, w=256):
    """
    Converts annotations in degrees to Kent distribution parameters.

    Args:
        annotations (torch.Tensor): Tensor with annotation data.
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        torch.Tensor: Kent distribution parameters.
    """
    
    
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)  # Convert to batch of size 1

    kent_params = []
    for idx, annotation in enumerate(annotations):
        eta, alpha, psi = annotation[0]/360*2*np.pi, annotation[1]/180*np.pi, 0
        #print(eta, alpha, psi)
        Xs = sample_from_annotation_deg(annotation, (h, w))
        S_torch, xbar_torch = get_me_matrix_torch(Xs)
        kappa, beta = kent_me_matrix_torch(S_torch, xbar_torch)

        k_torch = torch.tensor([eta, alpha, psi, kappa, beta])
        
        kent_params.append(k_torch)
    
    return torch.stack(kent_params)
'''
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
    
    # Normalize input data
    data_x = annotations[:, 0] / 360.0  # Normalize longitude
    data_y = annotations[:, 1] / 180.0  # Normalize latitude
    data_fov_h = annotations[:, 2]#.clamp(1.0, 179.0)  # Clamp horizontal FOV
    data_fov_v = annotations[:, 3]#.clamp(1.0, 179.0)  # Clamp vertical FOV
    
    #pdb.set_trace()

    # Compute angles
    eta = data_x * 2 * np.pi
    alpha = data_y * np.pi
    psi = torch.zeros_like(eta)

    varphi = (torch.deg2rad(data_fov_h) ** 2) / 12 + epsilon  # Horizontal variance
    vartheta = (torch.deg2rad(data_fov_v) ** 2) / 12 + epsilon  # Vertical variance

    kappa = 0.5 * (1 / varphi + 1 / vartheta)
    beta = torch.abs(0.25 * (1 / vartheta - 1 / varphi))

    return torch.stack([eta, alpha, psi, kappa, beta], dim=1)
'''
def compare_methods(annotations, h, w, threshold=0.2):
    print("Annotations:\n", annotations)
    
    # Kent params using deg2kent_single
    params_single = deg2kent_sampling(annotations, h, w)
    print("\nKent parameters (Single method):\n", params_single[:, 3:])
    
    # Kent params using bfov_to_kent
    params_bfov = bfov_to_kent(annotations)
    print("\nKent parameters (BFOV method):\n", params_bfov[:, 3:])
    
    # Compare both results
    difference = torch.abs(params_single - params_bfov) / (torch.abs(params_single) + 1e-6) * 100  # Percentage difference
    print("\nPercentage difference between methods:\n", difference[:, :])
    
    # Find significant differences
    significant_diff = difference[difference > threshold * 100]
    if significant_diff.numel() > 0:
        print("\nSignificant differences (greater than threshold):")
        for idx in range(difference.size(0)):
            if (difference[idx] > threshold * 100).any():
                print(f"Annotation: {annotations[idx]}, Percentage Difference: {difference[idx]}")
    else:
        print("\nNo significant differences found.")


if __name__=="__main__":
    # Generate annotations with specified ranges
    num_samples = 10000  # Number of samples to generate
    annotations = torch.tensor([[torch.rand(1).item() * 360,  # First parameter: 0 to 360
                                torch.rand(1).item() * 180,  # Second parameter: 0 to 180
                                torch.rand(1).item() * 100 + 1,  # Third parameter: 1 to 60
                                torch.rand(1).item() * 100 + 1]  # Fourth parameter: 1 to 60
                                for _ in range(num_samples)], dtype=torch.float32)


    #annotations = torch.tensor([100.1204,  68.6243,  47.8202,  38.8385])
    # Compare methods
    compare_methods(annotations, h=480, w=960)