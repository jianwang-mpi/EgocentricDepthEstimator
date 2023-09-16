import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset.dataset_transforms import *
import os
import cv2
from tqdm import tqdm
from dataset.egocentric_utils import EgocentricSegmentationPreprocess


class RandomDepthDataset(Dataset):
    """Ego-centric dataset"""

    def __init__(self, data_dir, depth_wo_body=False, circle_crop=False):
        self.data_dir = data_dir
        self.depth_wo_body = depth_wo_body
        self.circle_crop = circle_crop

        self.data = []

        for scene_id in tqdm(os.listdir(self.data_dir)):
            scene_path = os.path.join(self.data_dir, scene_id)
            if os.path.isdir(scene_path) is False:
                continue
            for pose_id in os.listdir(scene_path):
                pose_path = os.path.join(scene_path, pose_id)
                if os.path.exists(os.path.join(pose_path, 'metadata.npy')):
                    img_dir = os.path.join(pose_path, 'img')
                    depth_dir = os.path.join(pose_path, 'depth')
                    if self.depth_wo_body is True:
                        depth_nobody_dir = os.path.join(pose_path, 'depth_nobody')

                    for img_name in os.listdir(img_dir):
                        img_path = os.path.join(img_dir, img_name)
                        img_id = os.path.splitext(img_name)[0]
                        depth_path = os.path.join(depth_dir, img_id, 'Image0001.exr')
                        data_item = {'img': img_path, 'depth': depth_path}
                        if self.depth_wo_body is True:
                            depth_nobody_path = os.path.join(depth_nobody_dir, img_id, 'Image0001.exr')
                            data_item['depth_nobody'] = depth_nobody_path

                        self.data.append(data_item)

    def __len__(self):
        return len(self.data)

    def get_mask(self, depth):
        mask = depth > 999.
        mask = mask.astype(np.float)

        img_h, img_w, _ = depth.shape

        circle_mask = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, center=(img_w // 2, img_h // 2),
                                 radius=int(360 / 1024 * img_h * np.sqrt(2)),
                                 color=(255, 255, 255), thickness=-1)
        circle_mask = (circle_mask > 0).astype(np.uint8)
        return 1 - mask * circle_mask

    def get_circle_mask(self, img_h, img_w):
        circle_mask = np.zeros(shape=(img_h, img_w, 3), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, center=(img_w // 2, img_h // 2),
                                 radius=int(360 / 1024 * img_h * np.sqrt(2)),
                                 color=(255, 255, 255), thickness=-1)
        circle_mask = (circle_mask > 0).astype(np.uint8)
        return circle_mask

    def __getitem__(self, index):

        depth = self.data[index]['depth']
        depth = cv2.imread(depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth[:, :, 0]

        img_h, img_w = depth.shape
        circle_mask = self.get_circle_mask(img_h=img_h, img_w=img_w)
        circle_mask_one_channel = circle_mask[:, :, 0]
        # seg depth
        if self.circle_crop:
            background = np.ones_like(depth) * 1e10
            depth = depth * circle_mask_one_channel + background * (1 - circle_mask_one_channel)

        if self.depth_wo_body:
            depth_wo_body = self.data[index]['depth_nobody']
            depth_wo_body = cv2.imread(depth_wo_body, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_wo_body = depth_wo_body[:, :, 0]

            if self.circle_crop:
                background = np.ones_like(depth) * 1e10
                depth_wo_body = depth_wo_body * circle_mask_one_channel + background * (1 - circle_mask_one_channel)

        cut_side = int(128 / 1280 * img_w)

        depth = depth[:, cut_side: -cut_side]
        depth[depth > 10] = 10

        # cv2.imshow('depth', depth / 10)
        # cv2.waitKey(0)
        depth = cv2.resize(depth, dsize=(128, 128))

        depth = torch.from_numpy(depth).float()
        depth = depth.unsqueeze(0)
        sample = {'depth': depth}

        if self.depth_wo_body:
            depth_wo_body = depth_wo_body[:, cut_side:-cut_side]
            depth_wo_body[depth_wo_body > 10] = 10
            depth_wo_body = cv2.resize(depth_wo_body, dsize=(128, 128))
            # cv2.imshow('depth', depth_wo_body / 10)
            # cv2.waitKey(0)
            depth_wo_body = torch.from_numpy(depth_wo_body).float()
            depth_wo_body = depth_wo_body.unsqueeze(0)
            sample['depth_nobody'] = depth_wo_body

        return sample


def getTrainingData(training_data_dir=r'/HPS/EgoSyn/work/synthetic/depth_matterport_single_image',
                    batch_size=64, depth_wo_body=True,
                    circle_crop=True):
    dataset = RandomDepthDataset(data_dir=training_data_dir,
                                 depth_wo_body=depth_wo_body,
                                 circle_crop=circle_crop)

    dataloader_training = DataLoader(dataset, batch_size,
                                     shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    return dataloader_training


if __name__ == '__main__':
    dataset = RandomDepthDataset(data_dir=r'\\winfs-inf\CT\EgoMocap\work\synthetic\depth_matterport_single_no_body',
                         depth_wo_body=True,
                         circle_crop=False)

    data = dataset[4000]
    depth = data['depth']
    depth = data['depth_nobody']
