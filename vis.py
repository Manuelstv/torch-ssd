from model import SSD300
import cv2
import random
import numpy as np
import json
from numpy.linalg import norm
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
    #img = plot_circles(img, kernel, color, 0.25)

    img = Plotting.plotEquirectangular(img, kernel, color, h, w)
    img = np.roll(img, w - t, axis=1)

    #pdb.set_trace()

    return img

if __name__ == "__main__":

    priors = SSD300(n_classes=5).create_prior_boxes

    image = imread('/home/mstveras/OmniNet/dataset/train/images/7l2b0.jpg')
    h, w = image.shape[:2]
    #with open('/home/mstveras/OmniNet/dataset/train/labels/7l2b0.json', 'r') as f:
    #    data = json.load(f)
    #boxes = data['boxes']

    boxes = priors()
    
    #classes = data['class']
    #color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 35: (64, 0, 0), 36: (0, 64, 0)}
    #for i in range(len(boxes)):
    for i in range(3000,3050):

        k = random.randint(0, 8500)
        box = boxes[k].cpu()
        u00, v00, a_lat1, a_long1 = box[0]*(w), box[1]*(h), box[2]*45, box[3]*45
        a_long = np.radians(a_long1)
        a_lat = np.radians(a_lat1)
        #color = color_map.get(classes[i], (255, 255, 255))
        color = (0,255,0)
        image = plot_bfov(image, v00, u00, a_long, a_lat, color, h, w)
    cv2.imwrite('final_image.png', image)
