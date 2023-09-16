import open3d
import json
import os
import pickle
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

# import utils.data_transforms as transforms
from dataset.data_transforms_new import Normalize, ToTensor
from config_da import args, consts
from tqdm import tqdm
import random
from dataset.dataset_utils import generate_target

class FinetuneWildDataset_with_seg(Dataset):
    """
    {'image_path': image_path_list[i],
                  'global_pose': global_optimized_pose_seq[i],
                  'gt_pose': gt_pose_seq[i],
                  'joints_2d': joints_2d_list[i],
                  'depth': depth_list[i]}
    """

    def __init__(self, root_data_path, with_seg=False, seg_width=256, is_train=True, local_machine=False):
        """

        :param root_data_path:
        :param is_train:
        :param is_zoom:
        :param local_machine:
        :param use_estimated_pose: use estimated pose as the pseudo ground truth, default: False
        """
        self.root_data_path = root_data_path
        self.with_seg = with_seg
        self.seg_width = seg_width
        # get data
        self.data = []
        # identity_name_list_old = ['ayush', 'ayush_new', 'binchen', 'chao', 'chao_new',
        #                           'kripa', 'kripa_new', 'lingjie', 'lingjie_new', 'mohamed', 'soshi_new']
        identity_name_list = ['ayush', 'ayush_new', 'binchen', 'chao', 'chao_new',
                              'kripa', 'kripa_new', 'mohamed', 'lingjie', 'lingjie_new',
                              'soshi_new', 'mengyu_new', 'zhili_new']
        for identity_name in identity_name_list:
            identity_path = os.path.join(self.root_data_path, identity_name)
            self.data.extend(self.get_real_identity_data(identity_path))

        self.normalize = Normalize(mean=consts.img.mean, std=consts.img.std)
        self.to_tensor = ToTensor()

        self.is_train = is_train
        self.local_machine = local_machine

    def get_real_data_single_seq(self, seq_dir):
        pkl_path = os.path.join(seq_dir, 'pseudo_gt.pkl')
        with open(pkl_path, 'rb') as f:
            seq_data = pickle.load(f)
        return seq_data

    def get_real_identity_data(self, identity_path):
        identity_data = []
        for seq_name in os.listdir(identity_path):
            seq_dir = os.path.join(identity_path, seq_name)
            # if 'rountunda' in seq_dir:
            #     continue
            if os.path.isdir(seq_dir):
                seq_data = self.get_real_data_single_seq(seq_dir)
                identity_data.extend(seq_data)

        return identity_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_i = self.data[index]

        image_path = data_i['image_path']
        out_data_item = {}
        if self.local_machine:
            image_path = image_path.replace('/HPS', 'X:')
            out_data_item['img_path'] = image_path
        else:
            image_path = image_path.replace('X:', '/HPS')

        img = cv2.imread(image_path)
        # bgr to rgb!!!
        img = img[:, :, ::-1]
        img = img[:, 128: -128, :]

        try:
            heatmap = generate_target(joints_2d)
        except Exception as e:
            print("error!")
            print('image_path: {}'.format(image_path))
            print(e)
            return self.__getitem__(index + 1)

        # data augmentation
        img = cv2.resize(img, dsize=(256, 256)) / 255.
        img = self.normalize(img)
        img_torch = self.to_tensor(img)
        out_data_item['img'] = img_torch

        # get segmentations
        # get seg file from image path
        if self.with_seg is True:
            image_dir, image_name_with_ext = os.path.split(image_path)
            image_name = os.path.splitext(image_name_with_ext)[0]
            seq_dir = os.path.split(image_dir)[0]
            seg_dir = os.path.join(seq_dir, 'segs')
            seg_path = os.path.join(seg_dir, '{}.pkl'.format(image_name))

            with open(seg_path, 'rb') as f:
                seg_data = pickle.load(f)
            seg = np.round(seg_data)
            seg = cv2.resize(seg, (self.seg_width, self.seg_width), interpolation=cv2.INTER_NEAREST)

            # cv2.imshow('seg', seg)
            # cv2.waitKey(0)

            seg = torch.from_numpy(seg).float()
            seg = seg.unsqueeze(0)
            out_data_item['seg'] = seg

        return out_data_item



if __name__ == '__main__':
    dataset = FinetuneWildDataset_with_seg(root_data_path=r'X:\Mo2Cap2Plus1\static00\ExternalEgo\External_camera_all',
                                  local_machine=True, with_seg=True)
    print(len(dataset))
    # 3150
    data_item = dataset[18050]
    image_path = data_item['img_path']
    img_torch = data_item['img']
    print(image_path)

    img_np = img_torch.numpy()
    img_np = img_np.transpose((1, 2, 0))
    img_np = img_np * consts.img.std + consts.img.mean

    cv2.imshow('img', img_np[:, :, ::-1])
    cv2.waitKey(0)
