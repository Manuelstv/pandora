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

# --- Helper Functions ---
def plot_circles(img, arr, color, alpha=0.4):
    """
    Plots circles with transparency on an image.

    Args:
        img: The image to draw circles on.
        arr: An array containing the center coordinates of the circles.
        color: The color of the circles (B, G, R).
        alpha: The transparency level (0.0 - 1.0), where 0.0 is fully transparent and 1.0 is fully opaque.

    Returns:
        The image with transparent circles drawn on it.
    """

    overlay = img.copy()

    for center in arr:
        cv2.circle(overlay, center, 10, (*color, int(255 * alpha)), -1)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# Helper function to calculate the projected coordinates of a point in 3D space
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
    """Plots a binocular field of view (BFOV) overlay on an equirectangular image.

    This function takes an equirectangular image and parameters defining the 
    position and size of a BFOV. It calculates the projection of a hemispherical 
    grid onto the image plane and marks the resulting points with circles.

    Args:
        image: The equirectangular image as a NumPy array (H x W x 3).
        v00: The vertical coordinate of the BFOV center in the image (pixels).
        u00: The horizontal coordinate of the BFOV center in the image (pixels).
        fov_lat: The latitude angle of the BFOV (radians).
        fov_long: The longitude angle of the BFOV (radians).
        color: The RGB color of the BFOV circles (e.g., (255, 0, 0) for red).
        h: The height of the image (pixels).
        w: The width of the image (pixels).

    Returns:
        The modified image with the BFOV overlay as a NumPy array (H x W x 3).
    """

    # Shift the image to center the BFOV
    t = int(w // 2 - u00)
    u00 += t
    image = np.roll(image, t, axis=1)

    # Calculate angles and projection parameters
    phi00 = (u00 - w / 2) * (2 * np.pi / w)
    theta00 = -(v00 - h / 2) * (np.pi / h)
    r = 10
    d_lat = r / (2 * np.tan(fov_lat / 2))
    d_long = r / (2 * np.tan(fov_long / 2))

    # Create rotation matrix
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))

    # Create grid of points
    p = np.array([[i * d_lat / d_long, j, d_lat] 
                  for i in range(-(r - 1) // 2, (r + 1) // 2 + 1)
                  for j in range(-(r - 1) // 2, (r + 1) // 2 + 1)])

    # Project points and create kernel
    kernel = np.array([project_point(point, R, w, h) for point in p]).astype(np.int32)

    # Ensure color is in BGR format for OpenCV
    color = (color[2], color[1], color[0])  # Convert RGB to BGR

    # Plot circles and equirectangular lines
    image = plot_circles(image, kernel, color)
    #image = plotEquirectangular(image, kernel, color)

    # Shift the image back
    image = np.roll(image, w - t, axis=1)

    return image

# --- Projection Functions (Kent/FB5) ---

def projectEquirectangular2Sphere(u, w, h):
    phi = u[:, 1] * (np.pi / float(h))     # Latitude angle (phi)
    theta = u[:, 0] * (2. * np.pi / float(w))  # Longitude angle (theta)
    sinphi = np.sin(phi)
    return np.vstack([sinphi * np.cos(theta), sinphi * np.sin(theta), np.cos(phi)]).T  # Cartesian coordinates (x, y, z)

# --- Angle to Rotation Matrix ---

def angle2Gamma(alpha, eta, psi):
    gamma_1 = asarray([cos(alpha), sin(alpha) * cos(eta), sin(alpha) * sin(eta)])  # Unit mean axis
    gamma_2 = asarray([-cos(psi) * sin(alpha), cos(psi) * cos(alpha) * cos(eta) - sin(psi) * sin(eta),
                       cos(psi) * cos(alpha) * sin(eta) + sin(psi) * cos(eta)])  # Unit major axis
    gamma_3 = asarray([sin(psi) * sin(alpha), -sin(psi) * cos(alpha) * cos(eta) - cos(psi) * sin(eta),
                       -sin(psi) * cos(alpha) * sin(eta) + cos(psi) * cos(eta)])  # Unit minor axis
    return asarray([gamma_1, gamma_2, gamma_3])

def projectSphere2Equirectangular(x, w, h):
    phi = np.arccos(np.clip(x[:, 2], -1, 1))     # Latitude angle (phi)
    theta = np.arctan2(x[:, 1], x[:, 0])          # Longitude angle (theta)
    theta[theta < 0] += 2 * np.pi         # Ensure theta is in [0, 2*pi)
    return np.vstack([theta * float(w) / (2. * np.pi), phi * float(h) / np.pi])


# --- Kent (FB5) Distribution ---
def FB5(Theta, X):
    def __c(kappa, beta):
        epsilon = 1e-6
        
        term1 = kappa - 2 * beta
        term2 = kappa + 2 * beta
        
        denominator = (term1 * term2 + epsilon)**(-0.5)  # Add epsilon to avoid division by zero
        
        #result = 2 * np.pi * exp_kappa * denominator
        result = np.log(2*np.pi) + kappa - np.log(denominator)*0.5
        return result

    kappa, beta, Q = Theta           # Unpack parameters
    gamma_1, gamma_2, gamma_3 = Q    # Unpack rotation matrix

    # Ensure parameters are valid
    assert beta >= 0 and beta < kappa / 2
    #assert isclose(dot(gamma_1, gamma_2), 0) and isclose(dot(gamma_2, gamma_3), 0)

    # Calculate probabilities for all points in X
    dot_products = np.dot(X, np.array([gamma_1, gamma_2, gamma_3]).T)  # Shape: (N, 3)
    
    return (2 * np.pi * np.exp(__c(kappa, beta))) ** -1 * np.exp(
        kappa * dot_products[:, 0] + beta * (dot_products[:, 1] ** 2 - dot_products[:, 2] ** 2)
    )

def main():
    image = cv2.imread('datasets/360INDOOR/images/6831370124_0615cf0411_f.jpg')
    if image is None:
        raise ValueError("Could not read the image file. Please check the path.")
        
    h, w = image.shape[:2]
    
    #with open('6831370124_0615cf0411_f.json', 'r') as f:
    with open('mock.json', 'r') as f:
        data = json.load(f)
    
    boxes = data['boxes']
    classes = data['class']
    
    color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 
                 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 
                 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 
                 35: (64, 0, 0), 36: (0, 64, 0)}
    
    original_for_heatmap = image.copy()

    transform  = SphBox2KentTransform()
    
    for i in range(len(boxes)):
        box = boxes[i]
        print(box)
        #u00, v00, _, _, a_long, a_lat, class_name = box
        u00, v00, a_long, a_lat = box

        a_lat = np.radians(a_lat)
        a_long = np.radians(a_long)
        
        color = color_map.get(classes[i], (255, 255, 255))
        color_bgr = (color[2], color[1], color[0])
        image = plot_bfov(image, v00, u00, a_lat, a_long, color_bgr, h, w)
        kent = transform(torch.tensor(box))
        #pdb.set_trace()
    
    cv2.imwrite('bfov_image.png', image)

    v, u = np.mgrid[0:h:1, 0:w:1]
    X = projectEquirectangular2Sphere(np.vstack((u.reshape(-1), v.reshape(-1))).T, w, h)

    #pdb.set_trace()

    #psi, alpha, eta, kappa, beta = 1.5135277107138325, -2.989421759431538, 0.6378340180954485, 620.7227651655078, 3.410605131648481e-13
    #psi, alpha, eta, kappa, beta = 2.4347,  -2.0774,   1.5573, 215.3134,  30.2054
    psi, alpha, eta, kappa, beta = kent.detach().numpy()[0]

    Q = angle2Gamma(alpha, eta, psi)
    theta = (kappa, beta, Q)

    kent = FB5(theta, X)

    kent_norm = (kent - kent.min()) / (kent.max() - kent.min())
    
    # 2. Apply gamma correction to enhance visibility
    gamma = 0.5  # Adjust this value to control brightness (lower = brighter)
    kent_gamma = np.power(kent_norm, gamma)
    
    # 3. Convert to uint8 [0, 255]
    kent_uint8 = (kent_gamma * 255).astype(np.uint8)
    kent_image = kent_uint8.reshape((h, w))
    
    heatmap_raw = cv2.applyColorMap(kent_image, cv2.COLORMAP_HOT)
    heatmap = heatmap_raw.astype(np.float32) / 255.0
    heatmap = np.clip(heatmap * 1.2, 0, 1)  # Increase intensity
    heatmap = (heatmap * 255).astype(np.uint8)
    
    original_float = original_for_heatmap.astype(np.float32) / 255.0
    heatmap_float = heatmap.astype(np.float32) / 255.0
    
    alpha = np.mean(heatmap_float, axis=2) * 0.5  # Adjust 0.5 to control overlay strength
    alpha = np.stack([alpha] * 3, axis=2)
    
    blended = original_float * (1 - alpha) + heatmap_float * alpha

    image_with_heatmap = np.clip(blended * 255, 0, 255).astype(np.uint8)

    cv2.imwrite('bounding_kent.png', image_with_heatmap)

if __name__ == "__main__":
    main()