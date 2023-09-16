import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from demo_transform import *
import os
import cv2
import pickle
from dataset.egocentric_utils import EgocentricSegmentationPreprocess

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
        self.segmentation_process = EgocentricSegmentationPreprocess(img_h=1024, img_w=1280)

        # reading images from image dir
        image_dir = os.path.join(self.img_dir, 'imgs')
        seg_dir = os.path.join(self.img_dir, 'segs')
        for name in os.listdir(image_dir):
            if name.endswith('jpg') or name.endswith('png'):
                image_path = os.path.join(image_dir, name)
                image_name_wo_ext = os.path.splitext(name)[0]
                seg_path = os.path.join(seg_dir, '{}.pkl'.format(image_name_wo_ext))
                if os.path.exists(seg_path) is False:
                    seg_path = os.path.join(seg_dir, '{}.png'.format(image_name_wo_ext))
                self.data.append((image_path, seg_path))

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
        img_path, seg_path = self.data[idx]
        image = Image.open(img_path)

        image = image.convert("RGB")

        if seg_path.endswith('pkl'):
            with open(seg_path, 'rb') as f:
                data = pickle.load(f)
            # print(np.max(data))
            seg = np.round(data).astype(np.uint8)
            seg = cv2.resize(seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            seg = np.pad(seg, ((0, 0), (128, 128)))
            seg = 1 - seg
            seg = seg * 255

        else:
            seg = cv2.imread(seg_path)
            seg_label = self.segmentation_process.convert_segmentation_image_to_label(seg, mask_type='body')
            seg_label = seg_label.astype(np.uint8)
            seg = 1 - seg_label
            seg = seg * 255

        background = np.zeros(shape=(1024, 1280, 3)).astype(np.uint8)
        background[:, :, 0] = 255
        background = Image.fromarray(background, mode='RGB')
        seg = Image.fromarray(seg, mode='L')
        image = Image.composite(image, background, seg)
        image = image.resize((640, 512), resample=Image.BILINEAR)



        w, h = image.size

        image = image.crop((64, 0, w - 64, h))
        # image.show()

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

if __name__ == '__main__':
    nyu2_loader = readNyu2('../data/jian3', crop=False)

    for image, image_path in nyu2_loader:
        print(image_path)