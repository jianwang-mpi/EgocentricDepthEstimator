import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset.dataset_transforms import *
import os
import cv2
from tqdm import tqdm
from dataset.egocentric_utils import EgocentricSegmentationPreprocess


class EgoRealDepthDataset(Dataset):
    """Ego-centric dataset"""
    def __init__(self, data_dir, cleaned_data, green=True, with_seg=False, seg_width=128, depth_wo_body=False,
                 circle_crop=True):
        self.data_dir = data_dir
        self.cleaned_data = cleaned_data
        self.green = green
        self.depth_wo_body = depth_wo_body
        self.segmentation_process = EgocentricSegmentationPreprocess(img_h=1024, img_w=1280)
        self.circle_crop = circle_crop
        self.with_seg = with_seg
        self.seg_width = seg_width

        self.__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}


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
                    if self.with_seg is True:
                        seg_dir = os.path.join(pose_path, 'seg')
                    for img_name in os.listdir(img_dir):
                        img_path = os.path.join(img_dir, img_name)
                        img_id = os.path.splitext(img_name)[0]
                        depth_path = os.path.join(depth_dir, img_id, 'Image0001.exr')
                        data_item = {'img': img_path, 'depth': depth_path}
                        if self.depth_wo_body is True:
                            depth_nobody_path = os.path.join(depth_nobody_dir, img_id, 'Image0001.exr')
                            data_item['depth_nobody'] = depth_nobody_path
                        if self.with_seg is True:
                            seg_path = os.path.join(seg_dir, img_name)
                            data_item['seg'] = seg_path
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

    def ego_crop_center(self, img, depth):
        """
        crop a square image from the center
        :param img:
        :param depth:
        :return:
        """
        h, w = img.shape[0], img.shape[1]
        center_h = h // 2
        center_w = w // 2
        img = img[center_h - 180: center_h + 180, center_w - 180: center_w + 180, :]
        depth = depth[center_h - 180: center_h + 180, center_w - 180: center_w + 180]
        return img, depth

    def __getitem__(self, index):
        img = self.data[index]['img']

        img = cv2.imread(img)

        img_h, img_w, c = img.shape
        circle_mask = self.get_circle_mask(img_h=img_h, img_w=img_w)
        circle_mask_one_channel = circle_mask[:, :, 0]

        if img is None:
            print('img read error at index: {}, img: {}'.format(index, self.data[index]['img']))
            return self.__getitem__((index + 1) % self.__len__())
        # bgr to rgb
        img = img[:, :, ::-1]

        depth = self.data[index]['depth']
        depth = cv2.imread(depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.green:
            # green background
            hole_mask = self.get_mask(depth)
            img = img * hole_mask + (1 - hole_mask) * np.array([0, 255, 0])
            img = img.astype(np.uint8)
        if self.circle_crop:
            img = img * circle_mask
        depth = depth[:, :, 0]

        # seg depth
        if self.circle_crop:
            background = np.ones_like(depth) * 1e10
            depth = depth * circle_mask_one_channel + background * (1-circle_mask_one_channel)

        if self.with_seg:
            seg = self.data[index]['seg']
            seg = cv2.imread(seg)
            seg_label = self.segmentation_process.convert_segmentation_image_to_label(seg, mask_type='body').astype(np.float)
            # seg_label = np.repeat(seg_label[:, :, np.newaxis], 3, axis=2)
            if self.circle_crop:
                seg_label = seg_label * circle_mask[:, :, 0]

            # red_background = seg_label * np.array([255, 0, 0])
            # img = img * (1 - seg_label) + red_background
            # img = img.astype(np.uint8)

        if self.depth_wo_body:
            depth_wo_body = self.data[index]['depth_nobody']
            depth_wo_body = cv2.imread(depth_wo_body, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_wo_body = depth_wo_body[:, :, 0]

            if self.circle_crop:
                background = np.ones_like(depth) * 1e10
                depth_wo_body = depth_wo_body * circle_mask_one_channel + background * (1 - circle_mask_one_channel)

        cut_side = int(128 / 1280 * img_w)
        img = img[:, cut_side: -cut_side, :]
        img = cv2.resize(img, dsize=(256, 256))

        # cv2.imshow('img', img[:, :, ::-1])
        # cv2.waitKey(0)
        img = img / 255.
        img = (img - self.__imagenet_stats['mean']) / self.__imagenet_stats['std']
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        depth = depth[:, cut_side: -cut_side]
        depth[depth > 10] = 10
        depth = cv2.resize(depth, dsize=(128, 128))
        # cv2.imshow('depth', depth / 10)
        # cv2.waitKey(0)
        depth = torch.from_numpy(depth).float()
        depth = depth.unsqueeze(0)
        sample = {'image': img, 'depth': depth}

        if self.with_seg:
            seg_label = seg_label[:, cut_side: -cut_side]
            seg_label = cv2.resize(seg_label, dsize=(self.seg_width, self.seg_width), interpolation=cv2.INTER_NEAREST)
            # cv2.imshow('seg', seg_label)
            # cv2.waitKey(0)
            seg_label = torch.from_numpy(seg_label).float()
            seg_label = seg_label.unsqueeze(0)
            sample['seg'] = seg_label

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
                    batch_size=64, cleaned_data=False, green=True, with_seg=False, seg_width=128, depth_wo_body=True,
                    circle_crop=True):


    dataset = EgoDataset(data_dir=training_data_dir,
                         cleaned_data=cleaned_data,
                         green=green, with_seg=with_seg, seg_width=seg_width, depth_wo_body=depth_wo_body,
                         circle_crop=circle_crop)

    dataloader_training = DataLoader(dataset, batch_size,
                                     shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    return dataloader_training



if __name__ == '__main__':
    dataset = EgoDataset(data_dir=r'\\winfs-inf\CT\EgoMocap\work\synthetic\depth_matterport_single_no_body',
                         cleaned_data=False,
                         with_seg=False,
                         depth_wo_body=True,
                         circle_crop=False)

    data = dataset[14000]
    img = data['image']
    depth = data['depth']
