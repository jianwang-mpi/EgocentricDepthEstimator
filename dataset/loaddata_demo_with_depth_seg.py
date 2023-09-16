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


class TestDatasetwithSeg(Dataset):
    def __init__(self, data_dir, with_seg=False, seg_width=256, circle_crop=False):
        self.data_dir = data_dir
        self.data = []
        self.with_seg = with_seg
        self.seg_width = seg_width
        self.segmentation_process = EgocentricSegmentationPreprocess(img_h=1024, img_w=1280)
        self.circle_crop = circle_crop

        self.__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                                 'std': [0.229, 0.224, 0.225]}

        # reading images from image dir
        image_dir = os.path.join(self.data_dir, 'imgs')
        if self.with_seg:
            seg_dir = os.path.join(self.data_dir, 'segs')
        for name in sorted(os.listdir(image_dir)):
            if name.endswith('jpg') or name.endswith('png'):
                image_path = os.path.join(image_dir, name)
                data_item = {'img': image_path}
                image_name_wo_ext = os.path.splitext(name)[0]
                if self.with_seg:
                    seg_path = os.path.join(seg_dir, '{}.pkl'.format(image_name_wo_ext))
                    if os.path.exists(seg_path) is False:
                        seg_path = os.path.join(seg_dir, '{}.png'.format(image_name_wo_ext))
                    if os.path.exists(seg_path) is False:
                        seg_path = os.path.join(seg_dir, '{}.jpg'.format(image_name_wo_ext))
                    data_item['seg'] = seg_path
                self.data.append(data_item)

    def get_img_mask(img_width=1280, img_height=1024):
        radius = int(img_height / 2 - 30)
        mask = np.zeros(shape=[img_height, img_width, 3])
        cv2.circle(mask, center=(img_width // 2, img_height // 2), radius=radius, color=(255, 255, 255), thickness=-1)
        return mask / 255.

    def get_circle_mask(self, img_h, img_w):
        circle_mask = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, center=(img_w // 2, img_h // 2),
                                 radius=int(360 / 1024 * img_h * np.sqrt(2)),
                                 color=(255, 255, 255), thickness=-1)
        circle_mask = (circle_mask > 0).astype(np.uint8)
        return circle_mask

    def __getitem__(self, idx):


        data_item = self.data[idx]
        img_path = data_item['img']
        image = cv2.imread(img_path)
        image = image[:, :, ::-1]
        # bgr to rgb

        img_h, img_w, c = image.shape
        circle_mask = self.get_circle_mask(img_h=img_h, img_w=img_w)
        # circle_mask_one_channel = circle_mask[:, :, 0]

        if self.circle_crop:
            image = image * circle_mask

        image = cv2.resize(image, dsize=(640, 512), interpolation=cv2.INTER_LINEAR)

        img_h, img_w, c = image.shape

        cut_side = round(128 / 1280 * img_w)
        image = image[:, cut_side: -cut_side, :]
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        image = image / 255.
        # print(image)
        image = (image - self.__imagenet_stats['mean']) / self.__imagenet_stats['std']
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()

        out_data_item = {'img': image,
                         'img_path': img_path}

        if self.with_seg:
            seg_path = data_item['seg']
            if seg_path.endswith('pkl'):
                with open(seg_path, 'rb') as f:
                    data = pickle.load(f)
                seg = np.round(data)

                if self.circle_crop:
                    seg = seg * circle_mask[:, :, 0]

                seg = cv2.resize(seg, (self.seg_width, self.seg_width), interpolation=cv2.INTER_NEAREST)

            else:
                seg = cv2.imread(seg_path)
                seg = self.segmentation_process.convert_segmentation_image_to_label(seg, mask_type='body')
                seg_w = seg.shape[0]

                if self.circle_crop:
                    seg = seg * circle_mask[:, :, 0]

                cut_side = round(128 / 1280 * seg_w)
                seg = seg[:, cut_side: -cut_side]
                seg = cv2.resize(seg, dsize=(self.seg_width, self.seg_width), interpolation=cv2.INTER_NEAREST)
                seg = seg.astype(np.float)

            # cv2.imshow('seg', seg)
            # cv2.waitKey(0)

            seg = torch.from_numpy(seg).float()
            seg = seg.unsqueeze(0)
            out_data_item['seg'] = seg

        return out_data_item

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = TestDatasetwithSeg(data_dir='../data/synthetic_seg_1', with_seg=True, seg_width=256,
                                 circle_crop=True)

    data_item = dataset[1]

    print(data_item.keys())

    print(data_item['seg'].shape)
