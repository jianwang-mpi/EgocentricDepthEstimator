import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from demo_transform import *
import os
import cv2

class depthDataset(Dataset):
    def __init__(self, filename, transform=None):
        self.frame = filename
        self.transform = transform

    def __getitem__(self,idx):
        image = Image.open(self.frame)

        image = image.convert("RGB")

        image = image.resize((640, 512), resample=Image.BILINEAR)

        w, h = image.size

        image = image.crop((64, 0, w - 64, h))
        
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return int(1)


class MultipleImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, crop=False):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        self.crop = crop

        # reading images from image dir
        img_dir = os.path.join(self.img_dir, 'imgs')
        for name in sorted(os.listdir(img_dir)):
            if name.endswith('jpg') or name.endswith('png'):
                self.data.append(os.path.join(img_dir, name))

    def get_img_mask(img_width=1280, img_height=1024):
        radius = int(img_height / 2 - 30)
        mask = np.zeros(shape=[img_height, img_width, 3])
        cv2.circle(mask, center=(img_width // 2, img_height // 2), radius=radius, color=(255, 255, 255), thickness=-1)
        return mask / 255.

    def ego_crop_center(self, img):
        """
        crop a square image from the center
        :param img:
        :param depth:
        :return:
        """
        w, h = img.size
        center_h = h // 2
        center_w = w // 2
        # img = img[center_h - 180: center_h + 180, center_w - 180: center_w + 180, :]
        img = img.crop((center_w - 180, center_h - 180, center_w + 180, center_h + 180))
        # depth = depth[center_h - 180: center_h + 180, center_w - 180: center_w + 180]
        # return img, depth
        return img

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)

        image = image.convert("RGB")

        image = image.resize((640, 512), resample=Image.BILINEAR)

        w, h = image.size

        image = image.crop((64, 0, w - 64, h))


        if self.transform:
            image = self.transform(image)

        return image, img_path

    def __len__(self):
        return len(self.data)
     


def readNyu2(filename, crop=False):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    image_trans = MultipleImageDataset(filename,
                        transform=transforms.Compose([
                        Scale([256, 256]),
                        ToTensor(),                                
                        Normalize(__imagenet_stats['mean'],
                                 __imagenet_stats['std'])
                       ]), crop=crop)

    image = DataLoader(image_trans, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)


    return image
