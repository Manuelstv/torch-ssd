from model import SSD300
import cv2
import random
import numpy as np
from numpy import rad2deg
import json
from numpy.linalg import norm
import torch
from skimage.io import imread
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
    def plotEquirectangular(image, kernel, color, h,w):
        resized_image = cv2.resize(image, (h,w))
        kernel = kernel.astype(np.int32)
        hull = cv2.convexHull(kernel)
        cv2.polylines(resized_image, [hull], isClosed=True, color=color, thickness=2)
        return resized_image
    
def plot_circles(img, arr, color, transparency):
    """
    Draws transparent circles on an image at specified coordinates.

    Parameters:
    img (ndarray): The image on which to draw.
    arr (list): List of center coordinates for the circles.
    color (tuple): Color of the circles.
    transparency (float): Transparency of the circles.

    Returns:
    ndarray: The image with circles drawn.
    """
    overlay = img.copy()
    for point in arr:
        cv2.circle(overlay, point, 10, color, -1)
    
    cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img)
    return img


def plot_bfov(img, v00, u00, a_lat, a_long, color, h, w):
    """
    Plots a bounding field of view on an equirectangular img.

    Parameters:
    img (ndarray): The equirectangular img.
    v00, u00 (int): Pixel coordinates of the center of the field of view.
    a_lat, a_long (float): Angular size of the field of view in latitude and longitude.
    color (tuple): Color of the plot.
    h, w (int): Height and width of the img.

    Returns:
    ndarray: The img with the field of view plotted.
    """
    t = int(w//2 - u00)
    u00 += t
    img = np.roll(img, t, axis=1)

    #pdb.set_trace()

    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)
    r = 30
    d_lat = r / (2 * np.tan(a_lat / 2))
    d_long = r / (2 * np.tan(a_long / 2))
    p = [np.array([i * d_lat / d_long, j, d_lat]) for i in range(-(r - 1) // 2, (r + 1) // 2) for j in range(-(r - 1) // 2, (r + 1) // 2)]
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
    p = [np.dot(R, (point / norm(point))) for point in p]
    phi = [np.arctan2(point[0], point[2]) for point in p]
    theta = [np.arcsin(point[1]) for point in p]
    u = [(angle / (2 * np.pi) + 0.5) * w for angle in phi]
    v = [h - (-angle / np.pi + 0.5) * h for angle in theta]

    u = np.nan_to_num(u, nan=0)
    v = np.nan_to_num(v, nan=0)

    kernel = np.array([u, v], dtype=np.int32).T
    img = plot_circles(img, kernel, color, 0.25)

    img = Plotting.plotEquirectangular(img, kernel, color, h, w)
    img = np.roll(img, w - t, axis=1)

    return img

def plot_image(image, box2, target_height, target_width, color):
    # Convert image from PyTorch tensor to numpy array and resize
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (target_width, target_height))

    # Process each box to plot
    for box in box2.cpu():
        #pdb.set_trace()
        u00, v00 = ((rad2deg(box[0])/360)+0.5)*target_width, ((rad2deg(box[1])/180)+0.5)*target_height
        a_lat, a_long = (box[2]), (box[3])
        image = plot_bfov(image, v00, u00, a_long, a_lat, color, target_height, target_width)
    return image