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
from typing import List, Tuple, Dict, Optional, Union, Mapping
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
        return np.log(2 * np.pi) + kappa-0.5* np.log(term1 * term2 + epsilon)

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
        - normalization)

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
    # Softer normalization using percentile clipping

    p_low, p_high = np.percentile(distribution, [0, 97])
    dist_clip = np.clip(distribution, p_low, p_high)
    
    # Standard min-max normalization after clipping
    dist_norm = (dist_clip - dist_clip.min()) / (
        dist_clip.max() - dist_clip.min()
    )
    
    # Apply gamma correction
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

def load_image(image_path: Path) -> Tuple[NDArray, int, int]:
    """Load image and return it with dimensions."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]
    logger.info(f"Processing image {image_path.name} ({width}x{height})")
    return image, height, width

def create_sphere_points(height: int, width: int) -> NDArray:
    """Create spherical coordinate points for the entire image."""
    v, u = np.mgrid[0:height:1, 0:width:1]
    points = np.vstack((u.reshape(-1), v.reshape(-1))).T
    return project_equirectangular_to_sphere(points, width, height)

def process_box(
    box: BBox, 
    sphere_points: NDArray, 
    image: NDArray, 
    output_dir: Path,
    dimensions: Tuple[int, int],
    box_index: int
) -> NDArray:
    """Process a single bounding box and return its Kent distribution."""
    height, width = dimensions
    
    # Convert box to Kent parameters
    bbox_tensor = torch.tensor(
        [box.u00, box.v00, box.a_long, box.a_lat],
        dtype=torch.float32
    )
    
    transform = SphBox2KentTransform((height, width))
    kent_params = transform(bbox_tensor).detach().numpy()[0]

    #kent_params = np.array([3.133411407470703, 2.233476161956787, 0.0, 2.4742720127105713, 0.3991676867008209])

    params = KentParams(*kent_params)
    logger.info(f"Box {box_index} Kent parameters: {params}")
    
    # Compute distribution
    kent_values = compute_kent_distribution(params, sphere_points)
    kent_image = kent_values.reshape((height, width))
    
    # Save individual heatmap
    heatmap = create_heatmap(kent_image, image)
    cv2.imwrite(
        str(output_dir / f"kent_box_{box_index}_class_{box.category_id}.png"),
        heatmap
    )
    
    return kent_image

def load_category_mapping(annotation_path: Path) -> Dict[str, int]:
    """
    Create a mapping of category names to category IDs.
    
    Args:
        annotation_path: Path to COCO annotation file
    
    Returns:
        Dictionary mapping category names to their IDs
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    return {cat['name'].lower(): cat['id'] for cat in data['categories']}

def process_image(
    image_path: Path,
    annotation_path: Path,
    target_category: int =13,
    output_dir: Path = Path("output")
) -> None:
    """Process a 360-degree image and generate visualizations."""
    # Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load category mapping and resolve category ID
    if isinstance(target_category, str):
        category_mapping = load_category_mapping(annotation_path)
        target_category = category_mapping.get(target_category.lower())
        if target_category is None:
            available_categories = ', '.join(sorted(category_mapping.keys()))
            raise ValueError(
                f"Category '{target_category}' not found. Available categories: {available_categories}"
            )
    
    # Load data
    image, height, width = load_image(image_path)
    boxes = load_coco_annotations(image_path.name, annotation_path)
    logger.info(f"Found {len(boxes)} boxes")

    #boxes = [3.133411407470703, 2.233476161956787, 0.0, 2.4742720127105713, 0.3991676867008209]
    
    # Create coordinate grid
    sphere_points = create_sphere_points(height, width)
    
    # Process boxes
    combined_distribution = np.zeros((height, width), dtype=np.float32)
    for i, box in enumerate(boxes):
        if box.category_id != target_category:
            continue
        
        kent_image = process_box(
            box, 
            sphere_points, 
            image, 
            output_dir,
            (height, width),
            i
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

    # Available Categories:
    # ========================================
    # 1. toilet              2. board               3. mirror             
    # 4. bed                 5. potted plant        6. book              
    # 7. clock               8. phone               9. keyboard          
    # 10. tv                 11. fan                12. backpack          
    # 13. light              14. refrigerator       15. bathtub           
    # 16. wine glass         17. airconditioner     18. cabinet           
    # 19. sofa              20. bowl               21. sink              
    # 22. computer          23. cup                24. bottle            
    # 25. washer            26. chair              27. picture           
    # 28. window            29. door               30. heater            
    # 31. fireplace         32. mouse              33. oven              
    # 34. microwave         35. person             36. vase              
    # 37. table       
    
    try:
        # Now we can use the category name directly
        process_image(
            IMAGE_PATH, 
            ANNOTATION_PATH, 
            target_category="person",  # Can use string name instead of ID
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise