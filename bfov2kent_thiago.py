import cv2
import numpy as np
import torch
from numpy.linalg import norm
from scipy.special import jv as I_, gamma as G_
from numpy import *
from typing import Tuple
import json
from line_profiler import LineProfiler
import pdb
from sphdet.losses import SphBox2KentTransform

def plot_center_point(image, eta, alpha, color, w, h, radius=5):
    """
    Plot the center point of a Kent distribution on equirectangular image
    
    Args:
        image: Input image
        eta: longitude parameter
        alpha: colatitude parameter
        color: BGR color tuple
        w, h: image dimensions
        radius: radius of the circle
    """
    # Convert spherical coordinates to image coordinates
    u = (eta / (2 * np.pi) + 0.5) * w
    v = (-alpha / np.pi + 0.5) * h
    
    center = (int(u), int(v))
    cv2.circle(image, center, radius, color, -1)
    return image

class Rotation:
    @staticmethod
    def Rx(alpha):
        return np.asarray([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return np.asarray([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return np.asarray([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

class Plotting:
    @staticmethod
    def plotEquirectangular(image, kernel, color):
        resized_image = image
        kernel = kernel.astype(np.int32)
        hull = cv2.convexHull(kernel)
        cv2.polylines(resized_image,
                      [hull],
                      isClosed=True,
                      color=color,
                      thickness=2)
        return resized_image

def plot_circles(img, arr, color, alpha=0.4):
    overlay = img.copy()
    for center in arr:
        cv2.circle(overlay, center, 10, (*color, int(255 * alpha)), -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def project_point(point, R, w, h):
    point_rotated = np.dot(R, point / np.linalg.norm(point))
    phi = np.arctan2(point_rotated[0], point_rotated[2])
    theta = np.arcsin(point_rotated[1])
    u = (phi / (2 * np.pi) + 0.5) * w
    v = h - (-theta / np.pi + 0.5) * h
    return u, v

def plot_bfov(image: np.ndarray, v00: float, u00: float, 
              fov_lat: float, fov_long: float,
              color: Tuple[int, int, int], h: int, w: int) -> np.ndarray:
    t = int(w // 2 - u00)
    u00 += t
    image = np.roll(image, t, axis=1)

    phi00 = (u00 - w / 2) * (2 * np.pi / w)
    theta00 = -(v00 - h / 2) * (np.pi / h)
    r = 100
    d_lat = r / (2 * np.tan(fov_lat / 2))
    d_long = r / (2 * np.tan(fov_long / 2))

    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))

    p = np.array([[i * d_lat / d_long, j, d_lat] 
                  for i in range(-(r - 1) // 2, (r + 1) // 2 + 1)
                  for j in range(-(r - 1) // 2, (r + 1) // 2 + 1)])

    kernel = np.array([project_point(point, R, w, h) for point in p]).astype(np.int32)

    color = (color[2], color[1], color[0])
    image = plot_circles(image, kernel, color)
    image = np.roll(image, w - t, axis=1)
    return image

def projectEquirectangular2Sphere(u, w, h):
    """
    Convert equirectangular image coordinates to spherical coordinates
    u[:, 0]: x coordinate (0 to w) -> longitude (0 to 2π)
    u[:, 1]: y coordinate (0 to h) -> colatitude (0 to π)
    """
    theta = u[:, 0] * (2. * np.pi / float(w))  # Longitude [0, 2π]
    phi = u[:, 1] * (np.pi / float(h))         # Colatitude [0, π]
    
    # Convert to Cartesian coordinates using colatitude
    x = np.sin(phi) * np.cos(theta)  # Note: using sin(phi) for colatitude
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)                  # Note: using cos(phi) for colatitude
    return np.vstack([x, y, z]).T

def angle2Gamma(alpha, eta, psi):
    """
    Convert angles to rotation matrix
    alpha: colatitude [0, π]
    eta: longitude [0, 2π]
    psi: rotation [-π, π]
    """
    # Mean direction vector (using colatitude)
    gamma_1 = np.array([
        np.sin(alpha) * np.cos(eta),
        np.sin(alpha) * np.sin(eta),
        np.cos(alpha)
    ])
    
    # Create a tangent vector (perpendicular to gamma_1)
    # First, get a vector perpendicular to the radial direction
    temp = np.array([-np.sin(eta), np.cos(eta), 0])
    
    # Create gamma_2 (perpendicular to gamma_1)
    gamma_2 = np.cross(gamma_1, temp)
    gamma_2 = gamma_2 / np.linalg.norm(gamma_2)
    
    # Create gamma_3 to complete the orthonormal basis
    gamma_3 = np.cross(gamma_1, gamma_2)
    
    # Apply the rotation by psi in the gamma_2-gamma_3 plane
    gamma_2_new = np.cos(psi) * gamma_2 + np.sin(psi) * gamma_3
    gamma_3_new = -np.sin(psi) * gamma_2 + np.cos(psi) * gamma_3
    
    return np.array([gamma_1, gamma_2_new, gamma_3_new])

def projectSphere2Equirectangular(x, w, h):
    phi = np.arccos(np.clip(x[:, 2], -1, 1))     # colatitude
    theta = np.arctan2(x[:, 1], x[:, 0])         # longitude
    theta[theta < 0] += 2 * np.pi                 # ensure [0, 2π]
    return np.vstack([theta * float(w) / (2. * np.pi), phi * float(h) / np.pi])

def FB5(Theta, X):
    def __c(kappa, beta):
        epsilon = 1e-6
        term1 = kappa - 2 * beta
        term2 = kappa + 2 * beta
        denominator = (term1 * term2 + epsilon)**(-0.5)
        result = 2 * np.pi * np.exp(kappa) * denominator
        return result

    kappa, beta, Q = Theta
    gamma_1, gamma_2, gamma_3 = Q

    assert beta >= 0 and beta < kappa / 2

    dot_products = np.dot(X, np.array([gamma_1, gamma_2, gamma_3]).T)
    
    return (2 * np.pi * (__c(kappa, beta))) ** -1 * np.exp(
        kappa * dot_products[:, 0] + beta * (dot_products[:, 1] ** 2 - dot_products[:, 2] ** 2)
    )

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

def main():

    image_path = "datasets/360INDOOR/images/7fB4v.jpg"
    annotations_file = "datasets/360INDOOR/annotations/instances_val2017.json"
    
    # Get image name from path
    image_name = image_path.split('/')[-1]
    
    # Get boxes
    try:
        boxes = read_coco_json(image_name, annotations_file)
        print(f"Found {len(boxes)} boxes for image {image_name}")
        
        # Create list of bounding boxes
        bbox_list = [box['bbox'] for box in boxes]
        classes_list = [box['category_id'] for box in boxes]
    except Exception as e:
        print(f"Error reading annotations: {e}")
        return

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image file. Please check the path.")
        
    h, w, _ = image.shape
    print(f"Image dimensions: {w}x{h}")

    #bbox_list = [[box[0]/360*w, box[1]/180*h, box[2], box[3]] for box in bbox_list]

    data_x = np.array([[box[0]/360] for box in bbox_list])
    data_y = np.array([[box[1]/180] for box in bbox_list])
    data_fov_h = np.array([[box[2]]for box in bbox_list])
    data_fov_v = np.array([[box[3]]for box in bbox_list])

    phi, theta = 2 * np.pi * data_x, np.pi * data_y

    varphi = np.deg2rad(data_fov_h)**2/12
    vartheta = np.deg2rad(data_fov_v)**2/12

    kappa = .5*(1/varphi+1/vartheta)
    beta = .25*(1/vartheta-1/varphi)




    pdb.set_trace()
    boxes = bbox_list
    classes = classes_list
    
    
    color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 
                 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 
                 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 
                 35: (64, 0, 0), 36: (0, 64, 0)}
    
    transform = SphBox2KentTransform()
    
    # Create coordinate grid once (reusable for all boxes)
    v, u = np.mgrid[0:h:1, 0:w:1]
    points = np.vstack((u.reshape(-1), v.reshape(-1))).T
    X = projectEquirectangular2Sphere(points, w, h)
    
    # For visualization
    original_image = image.copy()
    combined_heatmap = np.zeros((h, w), dtype=np.float32)
    
    print(f"Processing {len(boxes)} boxes...")
    
    for i in range(len(boxes)):
        # Skip if category is not 35
        if classes[i] != 35:
            continue
            
        box = boxes[i]
        print(f"\nProcessing box {i}: {box}")
        u00, v00, a_long, a_lat = box
        bbox_tensor = torch.tensor([u00, v00, a_long, a_lat], dtype=torch.float32)
        kent = transform(bbox_tensor)

        eta, alpha, psi, kappa, beta = kent.detach().numpy()[0]
        print(f"Box {i} Kent parameters:")
        print(f"  eta (longitude): {eta:.4f}")
        print(f"  alpha (colatitude): {alpha:.4f}")
        print(f"  psi (rotation): {psi:.4f}")
        print(f"  kappa: {kappa:.4f}")
        print(f"  beta: {beta:.4f}")
            
        # Calculate Kent distribution
        Q = angle2Gamma(alpha, eta, psi)
        theta = (kappa, beta, Q)
        kent_values = FB5(theta, X)
        
        # Normalize and apply gamma correction
        kent_norm = (kent_values - kent_values.min()) / (kent_values.max() - kent_values.min())
        kent_gamma = np.power(kent_norm, 0.5)
        kent_image = kent_gamma.reshape((h, w))
        
        # Save individual heatmap
        color = color_map.get(classes[i], (255, 255, 255))
        heatmap_raw = cv2.applyColorMap((kent_image * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        # Blend with original image
        heatmap = heatmap_raw.astype(np.float32) / 255.0
        heatmap = np.clip(heatmap * 1.2, 0, 1)
        
        original_float = original_image.astype(np.float32) / 255.0
        alpha = np.mean(heatmap, axis=2) * 0.5
        alpha = np.stack([alpha] * 3, axis=2)
        
        blended = original_float * (1 - alpha) + heatmap * alpha
        blended_uint8 = np.clip(blended * 255, 0, 255).astype(np.uint8)
        
        # Save individual Kent plot
        cv2.imwrite(f'kent_box_{i}_class_{classes[i]}.png', blended_uint8)
        print(f"Saved kent_box_{i}_class_{classes[i]}.png")
        
        # Add to combined heatmap
        combined_heatmap += kent_image

    
    print("\nGenerating combined visualization...")
    
    # Normalize and visualize combined heatmap
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min())
    combined_heatmap_gamma = np.power(combined_heatmap, 0.5)
    combined_uint8 = (combined_heatmap_gamma * 255).astype(np.uint8)
    
    heatmap_raw = cv2.applyColorMap(combined_uint8, cv2.COLORMAP_HOT)
    heatmap = heatmap_raw.astype(np.float32) / 255.0
    heatmap = np.clip(heatmap * 1.2, 0, 1)
    
    original_float = original_image.astype(np.float32) / 255.0
    alpha = np.mean(heatmap, axis=2) * 0.5
    alpha = np.stack([alpha] * 3, axis=2)
    
    blended = original_float * (1 - alpha) + heatmap * alpha
    final_image = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    # Save combined heatmap
    cv2.imwrite('kent_combined.png', final_image)
    print("Saved kent_combined.png")
    
    # Draw BFOV circles on original image
    bfov_image = image.copy()
    for i in range(len(boxes)):
        # Skip if category is not 35
        if classes[i] != 35:
            continue
            
        box = boxes[i]
        u00, v00, a_long, a_lat = box
        a_lat = np.radians(a_lat)
        a_long = np.radians(a_long)
        color = color_map.get(classes[i], (255, 255, 255))
        color_bgr = (color[2], color[1], color[0])
        bfov_image = plot_bfov(bfov_image, v00, u00, a_lat, a_long, color_bgr, h, w)
    
    cv2.imwrite('bfov_image.png', bfov_image)
    print("Saved bfov_image.png")
    print("Processing complete!")

if __name__ == "__main__":
    main()