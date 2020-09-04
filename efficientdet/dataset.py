import os
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class EFIDataset(Dataset):
    def __init__(self, root_dir, csv_path, all_classes_dict_path, transform=None):

        self.root_dir = root_dir
        df = pd.read_csv(csv_path)
        columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
        df.columns = columns
        self.df = df
        self.all_classes_dict_path = all_classes_dict_path
        self.transform = transform

        self.image_ids = self.df.index.values
        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        self.classes = json.loads(open(self.all_classes_dict_path).read())

        # also load the reverse (label -> name)
        self.labels = {v: k for k, v in self.classes.items()}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        
        img_rel_name = self.df['filename'][image_index].split('latest/')[-1]
        img_path = os.path.abspath(os.path.join(self.root_dir, img_rel_name))
        img = np.array(Image.open(img_path).convert('RGB'))

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):

        index_filename = self.df['filename'][image_index]
        index_df = self.df[self.df['filename'] == index_filename]

        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(index_df) == 0:
            return annotations

        for idx, row in index_df.iterrows():

            xmin = max(0, row['xmin'])
            xmax = row['xmax']
            ymin = max(0, row['ymin'])
            ymax = row['ymax']

            class_ = row['class']
            label_ = self.classes[class_]


            annotation = np.zeros((1, 5))
            annotation[0, :4] = [xmin, ymin, xmax, ymax]
            annotation[0, 4] = label_

            annotations = np.append(annotations, annotation, axis=0)

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
