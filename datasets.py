import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from utils import transform

class PascalVOCDataset(Dataset):

    def __init__(self, data_folder, split, keep_difficult=False):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult
        self.split_dir = os.path.join(data_folder, self.split.lower())
        #self.image_dir = os.path.join(self.split_dir, 'images')
        self.image_dir = '/home/mstveras/ssd-360/train2/images'
        #self.annotation_dir = os.path.join(self.split_dir)
        self.annotation_dir = '/home/mstveras/ssd-360/train2/labels'
        print(self.annotation_dir)
        self.image_filenames = [os.path.join(self.image_dir, f) for f in sorted(os.listdir(self.image_dir)) if f.endswith('.jpg')]
        self.annotation_filenames = [os.path.join(self.annotation_dir, f) for f in sorted(os.listdir(self.annotation_dir)) if f.endswith('.xml')]
        assert len(self.image_filenames) == len(self.annotation_filenames)

    def __getitem__(self, i):
        image_filename = self.image_filenames[i]
        annotation_filename = self.annotation_filenames[i]
        image = Image.open(image_filename, mode='r').convert('RGB')

        tree = ET.parse(annotation_filename)
        root = tree.getroot()
        boxes = []
        labels = []
        difficulties = []

        label_mapping = {'airconditioner': 1, 'backpack': 2, 'bathtub': 3, 'bed': 4, 'board': 5, 'book': 6, 'bottle': 7, 'bowl': 8, 'cabinet': 9, 'chair': 10, 'clock': 11, 'computer': 12, 'cup': 13, 'door': 14, 'fan': 15, 'fireplace': 16, 'heater': 17, 'keyboard': 18, 'light': 19, 'microwave': 20, 'mirror': 21, 'mouse': 22, 'oven': 23, 'person': 24, 'phone': 25, 'picture': 26, 'potted plant': 27, 'refrigerator': 28, 'sink': 29, 'sofa': 30, 'table': 31, 'toilet': 32, 'tv': 33, 'vase': 34, 'washer': 35, 'window': 36, 'wine glass': 37}

        for obj in root.findall('object'):
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
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
