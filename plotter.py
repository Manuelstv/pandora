"""
360-degree image processing with Kent distribution visualization.

This module provides tools for processing 360-degree images, including:
- Coordinate transformations
- Kent distribution calculations
- Bounding box visualization
- COCO annotation parsing
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json
import logging
import numpy as np
import cv2
import torch
from numpy.typing import NDArray
from scipy.special import jv as I_, gamma as G_
from sphdet.losses import SphBox2KentTransform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KentParams:
    """Kent distribution parameters."""
    eta: float  # longitude
    alpha: float  # colatitude
    psi: float  # rotation
    kappa: float  # concentration
    beta: float  # ellipticity

@dataclass
class BBox:
    """Bounding box parameters."""
    u00: float  # center x
    v00: float  # center y
    a_long: float  # width
    a_lat: float  # height
    category_id: int

def create_rotation_matrix(angle: float, axis: str) -> NDArray:
    """
    Create rotation matrix around specified axis.
    
    Args:
        angle: Rotation angle in radians
        axis: 'x', 'y', or 'z'
    
    Returns:
        3x3 rotation matrix
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    if axis.lower() == 'x':
        return np.array([[1, 0, 0],
                        [0, cos_a, -sin_a],
                        [0, sin_a, cos_a]])
    elif axis.lower() == 'y':
        return np.array([[cos_a, 0, sin_a],
                        [0, 1, 0],
                        [-sin_a, 0, cos_a]])
    elif axis.lower() == 'z':
        return np.array([[cos_a, -sin_a, 0],
                        [sin_a, cos_a, 0],
                        [0, 0, 1]])
    else:
        raise ValueError(f"Invalid axis: {axis}. Use 'x', 'y', or 'z'.")

def project_equirectangular_to_sphere(points: NDArray, width: int, height: int) -> NDArray:
    """
    Convert equirectangular coordinates to spherical coordinates.
    
    Args:
        points: Nx2 array of (u, v) coordinates
        width: Image width
        height: Image height
    
    Returns:
        Nx3 array of (x, y, z) coordinates on unit sphere
    """
    theta = points[:, 0] * (2.0 * np.pi / width)  # Longitude [0, 2π]
    phi = points[:, 1] * (np.pi / height)         # Colatitude [0, π]
    
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

def project_sphere_to_equirectangular(points: NDArray, width: int, height: int) -> NDArray:
    """
    Convert spherical coordinates to equirectangular coordinates.
    
    Args:
        points: Nx3 array of (x, y, z) coordinates
        width: Image width
        height: Image height
    
    Returns:
        2xN array of (u, v) coordinates
    """
    phi = np.arccos(np.clip(points[:, 2], -1, 1))
    theta = np.arctan2(points[:, 1], points[:, 0])
    theta[theta < 0] += 2 * np.pi
    
    return np.vstack([
        theta * width / (2.0 * np.pi),
        phi * height / np.pi
    ])

def compute_kent_distribution(params: KentParams, points: NDArray) -> NDArray:
    """
    Compute Kent distribution values for given points.
    
    Args:
        params: Kent distribution parameters
        points: Nx3 array of unit vectors
    
    Returns:
        Array of Kent distribution values
    """
    def compute_log_normalization(kappa: float, beta: float, epsilon: float = 1e-6) -> float:
        term1 = kappa - 2 * beta
        term2 = kappa + 2 * beta
        return np.log(2 * np.pi) + kappa -0.5* np.log(term1 * term2 + epsilon)

    # Compute orthonormal basis
    gamma_1 = np.array([
        np.sin(params.alpha) * np.cos(params.eta),
        np.sin(params.alpha) * np.sin(params.eta),
        np.cos(params.alpha)
    ])
    
    temp = np.array([-np.sin(params.eta), np.cos(params.eta), 0])
    gamma_2 = np.cross(gamma_1, temp)
    gamma_2 /= np.linalg.norm(gamma_2)
    gamma_3 = np.cross(gamma_1, gamma_2)
    
    # Apply rotation
    cos_psi, sin_psi = np.cos(params.psi), np.sin(params.psi)
    gamma_2_new = cos_psi * gamma_2 + sin_psi * gamma_3
    gamma_3_new = -sin_psi * gamma_2 + cos_psi * gamma_3
    
    Q = np.array([gamma_1, gamma_2_new, gamma_3_new])
    
    # Compute distribution
    dot_products = points @ Q.T
    normalization = compute_log_normalization(params.kappa, params.beta)
    
    return np.exp(
        params.kappa * dot_products[:, 0] + 
        params.beta * (dot_products[:, 1] ** 2 - dot_products[:, 2] ** 2)
    ) / (2 * np.pi * normalization)

def load_coco_annotations(image_name: str, annotation_path: Path) -> List[BBox]:
    """
    Load COCO format annotations for specified image.
    
    Args:
        image_name: Image filename
        annotation_path: Path to COCO annotation file
    
    Returns:
        List of BBox objects
    """
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to load annotations: {e}")
        raise

    # Find image ID
    image_id = next(
        (img['id'] for img in data['images'] 
         if img['file_name'] == image_name), None
    )
    
    if image_id is None:
        raise ValueError(f"Image {image_name} not found in annotations")
    
    # Extract relevant boxes
    return [
        BBox(
            u00=ann['bbox'][0],
            v00=ann['bbox'][1],
            a_long=ann['bbox'][2],
            a_lat=ann['bbox'][3],
            category_id=ann['category_id']
        )
        for ann in data['annotations']
        if ann['image_id'] == image_id
    ]

def create_heatmap(
    distribution: NDArray,
    original_image: NDArray,
    gamma: float = 0.5,
    alpha: float = 0.5
) -> NDArray:
    """
    Create a heatmap visualization of the distribution.
    
    Args:
        distribution: 2D array of distribution values
        original_image: Original BGR image
        gamma: Gamma correction value
        alpha: Blending factor
    
    Returns:
        Blended heatmap image
    """
    # Normalize and apply gamma correction
    dist_norm = (distribution - distribution.min()) / (
        distribution.max() - distribution.min()
    )
    dist_gamma = np.power(dist_norm, gamma)
    
    # Convert to heatmap
    heatmap_raw = cv2.applyColorMap(
        (dist_gamma * 255).astype(np.uint8),
        cv2.COLORMAP_HOT
    )
    
    # Blend with original image
    heatmap = np.clip(heatmap_raw.astype(np.float32) / 255.0 * 1.2, 0, 1)
    original_float = original_image.astype(np.float32) / 255.0
    
    blend_alpha = np.mean(heatmap, axis=2) * alpha
    blend_alpha = np.stack([blend_alpha] * 3, axis=2)
    
    blended = original_float * (1 - blend_alpha) + heatmap * blend_alpha
    return np.clip(blended * 255, 0, 255).astype(np.uint8)

def process_image(
    image_path: Path,
    annotation_path: Path,
    target_category: int = 35,
    output_dir: Path = Path("output")
) -> None:
    """
    Process a 360-degree image and generate visualizations.
    
    Args:
        image_path: Path to input image
        annotation_path: Path to COCO annotations
        target_category: Category ID to process
        output_dir: Directory for output images
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image and annotations
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]
    logger.info(f"Processing image {image_path.name} ({width}x{height})")
    
    boxes = load_coco_annotations(image_path.name, annotation_path)
    logger.info(f"Found {len(boxes)} boxes")
    
    # Create coordinate grid
    v, u = np.mgrid[0:height:1, 0:width:1]
    points = np.vstack((u.reshape(-1), v.reshape(-1))).T
    sphere_points = project_equirectangular_to_sphere(points, width, height)
    
    # Process each box
    combined_distribution = np.zeros((height, width), dtype=np.float32)
    
    for i, box in enumerate(boxes):
        if box.category_id != target_category:
            continue
            
        # Convert box to Kent parameters
        bbox_tensor = torch.tensor(
            [box.u00, box.v00, box.a_long, box.a_lat],
            dtype=torch.float32
        )

        transform  = SphBox2KentTransform((height, width))

        kent_params = transform(bbox_tensor).detach().numpy()[0]
        
        params = KentParams(*kent_params)
        logger.info(f"Box {i} Kent parameters: {params}")
        
        # Compute distribution
        kent_values = compute_kent_distribution(params, sphere_points)
        kent_image = kent_values.reshape((height, width))
        
        # Save individual heatmap
        heatmap = create_heatmap(kent_image, image)
        cv2.imwrite(
            str(output_dir / f"kent_box_{i}_class_{box.category_id}.png"),
            heatmap
        )
        
        combined_distribution += kent_image
    
    # Save combined visualization
    combined_heatmap = create_heatmap(combined_distribution, image)
    cv2.imwrite(str(output_dir / "kent_combined.png"), combined_heatmap)
    logger.info("Processing complete!")

if __name__ == "__main__":
    # Configuration
    IMAGE_PATH = Path("datasets/360INDOOR/images/7fB4v.jpg")
    ANNOTATION_PATH = Path("datasets/360INDOOR/annotations/instances_val2017.json")
    OUTPUT_DIR = Path("output")
    
    try:
        process_image(IMAGE_PATH, ANNOTATION_PATH, output_dir=OUTPUT_DIR)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise