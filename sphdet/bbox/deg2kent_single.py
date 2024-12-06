import torch
import sys
import numpy as np
import torch.nn as nn
import pdb
from sphdet.bbox.kent_formator import kent_me_matrix_torch, get_me_matrix_torch
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

    r = 30
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
def deg2kent_single(annotations, h, w):
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


def bfov_to_kent(annotations):
    """
    Converts bounding field of view (BFOV) annotations to Kent distribution parameters.

    Args:
        annotations (torch.Tensor): [n, 4] tensor where each row is [x, y, fov_h, fov_v].

    Returns:
        torch.Tensor: [n, 5] tensor where each row is [eta, alpha, psi, kappa, beta].
    """
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)  # Convert to batch of size 1
    
    # Extract and normalize coordinates
    data_x = annotations[:, 0] / 360.0  # Normalize x coordinate (longitude)
    data_y = annotations[:, 1] / 180.0  # Normalize y coordinate (latitude)
    data_fov_h = annotations[:, 2]      # Horizontal FOV (degrees)
    data_fov_v = annotations[:, 3]      # Vertical FOV (degrees)
    
    # Compute angles phi and theta
    phi = 2 * np.pi * data_x            # Azimuthal angle (longitude)
    theta = np.pi * data_y              # Polar angle (latitude)
    
    # Convert FOV to radians and compute variances
    varphi = (torch.deg2rad(data_fov_h) ** 2) / 12  # Horizontal variance
    vartheta = (torch.deg2rad(data_fov_v) ** 2) / 12  # Vertical variance
    
    # Compute Kent distribution parameters
    kappa = 0.5 * (1 / varphi + 1 / vartheta)
    beta = 0.25 * (1 / vartheta - 1 / varphi)
    
    # Set angles for Kent distribution
    eta = phi  # Azimuthal mean direction
    alpha = theta  # Polar mean direction
    psi = torch.zeros_like(eta)  # Default twist angle (set to 0 for simplicity)
    
    # Stack parameters into [n, 5] tensor
    kent_dist = torch.stack([eta, alpha, psi, kappa, beta], dim=1)
    
    pdb.set_trace()
    
    return kent_dist
    


def deg2kent_sampling(annotations, h=980, w=1960):
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
        eta, alpha, psi = annotation[0]/w*2*np.pi, annotation[1]/h*np.pi, 0
        print(eta, alpha, psi)
        Xs = sample_from_annotation_deg(annotation, (h, w))
        S_torch, xbar_torch = get_me_matrix_torch(Xs)
        kappa, beta = kent_me_matrix_torch(S_torch, xbar_torch)

        k_torch = torch.tensor([eta, alpha, psi, kappa, beta])
        
        kent_params.append(k_torch)
    
    return torch.stack(kent_params)


def deg2kent_single_old(annotations, h, w):
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

    model_path = 'kent_mlp_model.pth'
    model, scaler = load_kent_model(model_path)

    kent_params = []
    for idx, annotation in enumerate(annotations):
        # Convert to radians while preserving gradients
        #if torch.all(torch.abs(annotation) < 1e-6):
        #    # Return zero parameters except kappa which should be 2500000
        #    print(annotation)
        #    zero_params = torch.zeros(5, device=annotation.device, dtype=annotation.dtype)
        #    zero_params[3] = 2500000.0  # Set kappa (4th parameter) to 2500000
        #    kent_params.append(zero_params)
        #    continue

        eta = (annotation[0]/360.0) * (2*torch.pi)
        alpha = (annotation[1]/180.0) * torch.pi
        psi = torch.tensor(0.0, device=annotation.device, dtype=annotation.dtype)
        
        #Xs = sample_from_annotation_deg(annotation, (h, w))
        #S_torch, xbar_torch = get_me_matrix_torch(Xs)
        #kappa, beta = kent_me_matrix_torch(S_torch, xbar_torch)

        annotation = annotation.reshape(-1, 4)

        kappa_predictions = predict_kappa(model, scaler, annotation)
        eta = eta.view(1)
        alpha = alpha.view(1)
        psi = psi.view(1)

        kappa, beta = kappa_predictions[:,0], kappa_predictions[:,1]
        
        #print(f"annotation {annotation}")
        #print(f"deg {kappa}, {beta}")
        #print(kappa2, beta2)

        #pdb.set_trace()
        
        k_torch = torch.stack([eta, alpha, psi, kappa, beta])
        
        kent_params.append(k_torch)
    
    return torch.stack(kent_params)


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