import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from numpy import deg2rad
from utils import transform

class PascalVOCDataset(Dataset):

    def __init__(self, split, keep_difficult=False, max_images=10, new_w = 600, new_h = 300):

        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST', 'VAL'}

        self.keep_difficult = keep_difficult
        self.new_h = new_h
        self.new_w = new_w

        # Base directory for datasets
        base_dir = '/home/mstveras/OmniNet/dataset'

        # Assign directory based on split
        if self.split == 'TRAIN':
            self.image_dir = os.path.join(base_dir, 'train/images2')
            self.annotation_dir = os.path.join(base_dir, 'train/labels2')
        elif self.split == 'VAL':
            self.image_dir = os.path.join(base_dir, 'val')
            self.annotation_dir = os.path.join(base_dir, 'val')
        elif self.split == 'TEST':
            self.image_dir = os.path.join(base_dir, 'test')
            self.annotation_dir = os.path.join(base_dir, 'test')
        self.split = split.upper()
        
        # Load all image files, sorting them to ensure that they are aligned
        self.image_filenames = [os.path.join(self.image_dir, f) for f in sorted(os.listdir(self.image_dir)) if f.endswith('.jpg')][:max_images]
        self.annotation_filenames = [os.path.join(self.annotation_dir, f) for f in sorted(os.listdir(self.annotation_dir)) if f.endswith('.xml')][:max_images]
        
        assert len(self.image_filenames) == len(self.annotation_filenames)

        for img_filename, ann_filename in zip(self.image_filenames, self.annotation_filenames):
            img_basename = os.path.splitext(img_filename)[0][-7:-3]
            ann_basename = os.path.splitext(ann_filename)[0][-7:-3]
            assert img_basename == ann_basename, f"File name mismatch: {img_filename} and {ann_filename}"

        # If max_images is set, limit the dataset size
        if max_images is not None:
            self.image_filenames = self.image_filenames[:max_images]
            self.annotation_filenames = self.annotation_filenames[:max_images]

    def __getitem__(self, i):
        image_filename = self.image_filenames[i]
        annotation_filename = self.annotation_filenames[i]
        image = Image.open(image_filename, mode='r').convert('RGB')

        w,h = image.size

        tree = ET.parse(annotation_filename)
        root = tree.getroot()
        boxes = []
        labels = []
        difficulties = []

        label_mapping = {'airconditioner': 1, 'backpack': 2, 'bathtub': 3, 'bed': 4, 'board': 5, 'book': 6, 'bottle': 7, 'bowl': 8, 'cabinet': 9, 'chair': 10, 'clock': 11, 'computer': 12, 'cup': 13, 'door': 14, 'fan': 15, 'fireplace': 16, 'heater': 17, 'keyboard': 18, 'light': 19, 'microwave': 20, 'mirror': 21, 'mouse': 22, 'oven': 23, 'person': 24, 'phone': 25, 'picture': 26, 'potted plant': 27, 'refrigerator': 28, 'sink': 29, 'sofa': 30, 'table': 31, 'toilet': 32, 'tv': 33, 'vase': 34, 'washer': 35, 'window': 36, 'wine glass': 37}

        for obj in root.findall('object'):
            #if obj.find('name').text == 'light':  # Check if the object is a person
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue

            if True:
                bbox = obj.find('bndbox')

                # Normalize pixel coordinates of center to [-1, 1]
                #xmin = deg2rad(180*2*(int(bbox.find('x_center').text)/w-1))
                #ymin = deg2rad(180*2*(int(bbox.find('y_center').text)/h-1))
                #xmax = (float(bbox.find('width').text))/180
                #ymax = (float(int(bbox.find('height').text)))/180

                xmin = int(bbox.find('x_center').text)/w
                ymin = int(bbox.find('y_center').text)/h

                xmin  = deg2rad(360*(xmin-0.5))
                ymin  = deg2rad(180*(ymin-0.5))

                xmax = deg2rad(float(bbox.find('width').text))
                ymax = deg2rad(float(int(bbox.find('height').text)))

                boxes.append([xmin,ymin,xmax,ymax])
                labels.append(label_mapping[obj.find('name').text])
                difficulties.append(difficult)

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        difficulties = torch.ByteTensor(difficulties)

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.image_filenames)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each