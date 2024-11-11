import cv2
import numpy as np
import math
from numpy.linalg import norm
from skimage.io import imread
from scipy.special import jv as I_, gamma as G_
from matplotlib import pyplot as plt
from numpy import *
from typing import Tuple
import json
from line_profiler import LineProfiler
import pdb

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


def deg_to_rad(degrees):
    return [math.radians(degree) for degree in degrees]

# --- Projection Functions (Kent/FB5) ---

def projectEquirectangular2Sphere(u, w, h):
    phi = u[:, 1] * (np.pi / float(h))     # Latitude angle (phi)
    theta = u[:, 0] * (2. * np.pi / float(w))  # Longitude angle (theta)
    sinphi = np.sin(phi)
    return np.vstack([sinphi * np.cos(theta), sinphi * np.sin(theta), np.cos(phi)]).T  # Cartesian coordinates (x, y, z)

def projectSphere2Equirectangular(x, w, h):
    phi = np.arccos(np.clip(x[:, 2], -1, 1))     # Latitude angle (phi)
    theta = np.arctan2(x[:, 1], x[:, 0])          # Longitude angle (theta)
    theta[theta < 0] += 2 * np.pi         # Ensure theta is in [0, 2*pi)
    return np.vstack([theta * float(w) / (2. * np.pi), phi * float(h) / np.pi])

# --- Angle to Rotation Matrix ---

def angle2Gamma(alpha, eta, psi):
    gamma_1 = asarray([cos(alpha), sin(alpha) * cos(eta), sin(alpha) * sin(eta)])  # Unit mean axis
    gamma_2 = asarray([-cos(psi) * sin(alpha), cos(psi) * cos(alpha) * cos(eta) - sin(psi) * sin(eta),
                       cos(psi) * cos(alpha) * sin(eta) + sin(psi) * cos(eta)])  # Unit major axis
    gamma_3 = asarray([sin(psi) * sin(alpha), -sin(psi) * cos(alpha) * cos(eta) - cos(psi) * sin(eta),
                       -sin(psi) * cos(alpha) * sin(eta) + cos(psi) * cos(eta)])  # Unit minor axis
    return asarray([gamma_1, gamma_2, gamma_3])

# --- Kent (FB5) Distribution ---
def FB5(Theta, X):
    '''def __c(kappa, beta, terms=10):
        su = 0
        for j in range(terms):
            su += G_(j + .5) / G_(j + 1) * beta ** (2 * j) * (2 / kappa) ** (2 * j + .5) * I_(2 * j + .5, kappa)
        return 2 * pi * su'''
    
    def __c(kappa, beta):
        epsilon = 1e-8  # Small value to avoid division by zero
        exp_kappa = np.exp(kappa)
        #pdb.set_trace()
        
        term1 = kappa - 2 * beta
        term2 = kappa + 2 * beta
        
        denominator = (term1 * term2 + epsilon)**(-0.5)  # Add epsilon to avoid division by zero
        
        result = 2 * np.pi * exp_kappa * denominator
        return result

    kappa, beta, Q = Theta           # Unpack parameters
    gamma_1, gamma_2, gamma_3 = Q    # Unpack rotation matrix

    # Ensure parameters are valid
    assert beta >= 0 and beta < kappa / 2
    assert isclose(dot(gamma_1, gamma_2), 0) and isclose(dot(gamma_2, gamma_3), 0)

    # Calculate probabilities for all points in X
    dot_products = np.dot(X, np.array([gamma_1, gamma_2, gamma_3]).T)  # Shape: (N, 3)
    return (2 * np.pi * __c(kappa, beta)) ** -1 * np.exp(
        kappa * dot_products[:, 0] + beta * (dot_products[:, 1] ** 2 - dot_products[:, 2] ** 2)
    )

def main():
    image = imread('datasets/360INDOOR/images/6831370124_0615cf0411_f.jpg')
    h, w = image.shape[:2]
    with open('6831370124_0615cf0411_f.json', 'r') as f:
        data = json.load(f)
    
    boxes = data['boxes']
    classes = data['class']
    
    color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 35: (64, 0, 0), 36: (0, 64, 0)}
    
    # Uncomment this block to profile the BFOV plotting
    
    for i in range(len(boxes)):
        box = boxes[i]
        u00, v00, _, _, a_lat1, a_long1, class_name = box
        a_lat = np.radians(a_long1)
        a_long = np.radians(a_lat1)
        color = color_map.get(classes[i], (255, 255, 255))
        image = plot_bfov(image, v00, u00, a_lat, a_long, color, h, w)
    
    cv2.imwrite('bfov_image.png', image)

    # --- 2. Kent (FB5) Visualization ---
    v, u = mgrid[0:h:1, 0:w:1]
    
    # Project pixel coordinates (u, v) from the equirectangular image to spherical coordinates (x, y, z)
    X = projectEquirectangular2Sphere(vstack((u.reshape(-1), v.reshape(-1))).T, w, h)

    # Calculate probabilities using the FB5 distribution for each point in spherical coordinates
    #psi, alpha, eta, kappa, beta = 1.7066047592157052, 1.909499284760046, -0.09826696971965836, 2482.6903491097846, 3.524291969370097e-12
    #psi, alpha, eta, kappa, beta = 1.8342319607677906, -0.7804894248762143, -1.570796326794897, 215.53367199419282, 30.240613209604945
    psi, alpha, eta, kappa, beta = 1.5135277107138325, -2.989421759431538, 0.6378340180954485, 620.7227651655078, 3.410605131648481e-13

    Q = angle2Gamma(alpha, eta, psi)
    theta = (kappa, beta, Q)

    kent = FB5(theta, X)

    # --- Convert Probability to Image ---
    kent_grayscale = (kent - kent.min()) / (kent.max() - kent.min()) * 255
    kent_image = kent_grayscale.reshape((h, w)).astype(np.uint8)
    heatmap = cv2.applyColorMap(kent_image, cv2.COLORMAP_HOT)
    # Ensure the original image is in the correct color format
    image_with_heatmap = cv2.addWeighted(image, 0.3, heatmap, 0.7, 0)  # Decrease image weight

    cv2.imwrite('bouding_kent.png', image_with_heatmap) 

if __name__ == "__main__":
    #profiler = LineProfiler()
    #profiler.add_function(main)  # Change this line to profile only the main function
    #profiler.run('main()')
    #profiler.print_stats()
    main()