import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset.dataset_transforms import *
import os
import cv2
from tqdm import tqdm
from dataset.egocentric_utils import EgocentricSegmentationPreprocess

class EgoTestDataset(Dataset):
    """
    Ego test dataset
    """

    def __init__(self, data_dir, transform, crop=False, total_length=1000, green=True):
        self.data_dir = data_dir
        self.transform = transform
        self.total_length = total_length
        self.crop = crop
        self.green = green
        self.data = []

        self.data = np.load(os.path.join(data_dir, 'test_data.npy'), allow_pickle=True)

        # np.random.shuffle(self.data)
        self.data = self.data[:self.total_length]

    def __len__(self):
        return len(self.data)

    def get_mask(self, depth):
        mask = depth > 999.
        mask = mask.astype(np.float)

        circle_mask = np.zeros(shape=(512, 640, 3), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, center=(320, 256), radius=int(180 * np.sqrt(2)),
                                 color=(255, 255, 255), thickness=-1)
        circle_mask = (circle_mask > 0).astype(np.float)
        return 1 - mask * circle_mask

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
        depth = self.data[index]['depth']

        img = cv2.imread(img)
        # bgr to rgb
        img = img[:, :, ::-1]
        depth = cv2.imread(depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.green:
            # green background
            hole_mask = self.get_mask(depth)
            img = img * hole_mask + (1 - hole_mask) * np.array([0, 255, 0])
            img = img.astype(np.uint8)
        depth = depth[:, :, 0]
        if self.crop:
            img, depth = self.ego_crop_center(img, depth)

        img = img[:, 64: -64, :]
        depth = depth[:, 64: -64]

        # do a simple normalization
        depth = depth / 10. * 255.
        depth[depth > 255] = 255
        depth = depth.astype(np.uint8)

        img = Image.fromarray(img)
        depth = Image.fromarray(depth)
        sample = {'image': img, 'depth': depth}

        sample = self.transform(sample)

        return sample


class EgoDataset(Dataset):
    """Ego-centric dataset"""

    def __init__(self, data_dir, cleaned_data, transform, crop=False, gray=False, green=True, depth_wo_body=False,
                 circle_crop=True):
        self.data_dir = data_dir
        self.transform = transform
        self.crop = crop
        self.gray = gray
        self.cleaned_data = cleaned_data
        self.green = green
        self.depth_wo_body = depth_wo_body
        self.segmentation_process = EgocentricSegmentationPreprocess(img_h=1024, img_w=1280)
        self.circle_crop = circle_crop


        self.data = []
        if self.cleaned_data is False:
            for scene_id in tqdm(os.listdir(self.data_dir)):
                scene_path = os.path.join(self.data_dir, scene_id)
                if os.path.isdir(scene_path) is False:
                    continue
                for pose_id in os.listdir(scene_path):
                    pose_path = os.path.join(scene_path, pose_id)
                    if os.path.exists(os.path.join(pose_path, 'metadata.npy')):
                        img_dir = os.path.join(pose_path, 'img')
                        if self.depth_wo_body is True:
                            depth_dir = os.path.join(pose_path, 'depth_nobody')
                        else:
                            depth_dir = os.path.join(pose_path, 'depth')
                        seg_dir = os.path.join(pose_path, 'seg')
                        for img_name in os.listdir(img_dir):
                            img_path = os.path.join(img_dir, img_name)
                            img_id = os.path.splitext(img_name)[0]
                            depth_path = os.path.join(depth_dir, img_id, 'Image0001.exr')
                            seg_path = os.path.join(seg_dir, img_name)
                            self.data.append({'img': img_path, 'depth': depth_path,
                                              'seg': seg_path})
        else:
            print('load cleaned dataset!')
            if self.depth_wo_body is True:
                train_data = os.path.join(self.data_dir, 'matterport_nobody_train.npy')
                test_data = os.path.join(self.data_dir, 'matterport_nobody_test.npy')
            else:
                train_data = os.path.join(self.data_dir, 'matterport_train.npy')
                test_data = os.path.join(self.data_dir, 'matterport_test.npy')
            self.data = np.load(train_data, allow_pickle=True)
            self.data = np.append(self.data, np.load(test_data, allow_pickle=True))
            print('loaded {} images!'.format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def get_mask(self, depth):
        mask = depth > 999.
        mask = mask.astype(np.float)

        circle_mask = np.zeros(shape=(1024, 1280, 3), dtype=np.uint8)
        circle_mask = cv2.circle(circle_mask, center=(640, 512), radius=int(360 * np.sqrt(2)),
                                 color=(255, 255, 255), thickness=-1)
        circle_mask = (circle_mask > 0).astype(np.float)
        return 1 - mask * circle_mask

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
        depth = self.data[index]['depth']
        seg = self.data[index]['seg']

        img = cv2.imread(img)
        seg = cv2.imread(seg)
        if img is None:
            print('img read error at index: {}, img: {}'.format(index, self.data[index]['img']))
            return self.__getitem__((index + 1) % self.__len__())
        if seg is None:
            raise Exception('seg read error at index: {}, seg: {}'.format(index, self.data[index]['seg']))
        # bgr to rgb
        img = img[:, :, ::-1]
        depth = cv2.imread(depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.green:
            # green background
            hole_mask = self.get_mask(depth)
            img = img * hole_mask + (1 - hole_mask) * np.array([0, 255, 0])
            img = img.astype(np.uint8)
        if self.circle_crop:
            img = self.segmentation_process.crop(img)
        depth = depth[:, :, 0]

        # seg depth
        if self.circle_crop:
            circle_mask = self.segmentation_process.circle_mask[:, :, 0]
            background = np.ones_like(depth) * 1e10
            depth = depth * circle_mask + background * (1-circle_mask)

        seg_label = self.segmentation_process.convert_segmentation_image_to_label(seg, mask_type='body')
        seg_label = np.repeat(seg_label[:, :, np.newaxis], 3, axis=2)
        if self.circle_crop:
            seg_label = self.segmentation_process.crop(seg_label)

        red_background = seg_label * np.array([255, 0, 0])
        img = img * (1 - seg_label) + red_background
        img = img.astype(np.uint8)

        assert img.shape[0] == 1024 and img.shape[1] == 1280
        assert depth.shape[0] == 1024 and depth.shape[1] == 1280
        img = img[:, 128: -128, :]
        depth = depth[:, 128: -128]

        # do a simple normalization
        depth = depth / 10. * 255.
        depth[depth > 255] = 255
        depth = depth.astype(np.uint8)

        img = Image.fromarray(img)
        depth = Image.fromarray(depth)
        sample = {'image': img, 'depth': depth}

        sample = self.transform(sample)

        return sample


def getTrainingData(training_data_dir=r'/HPS/EgoSyn/work/synthetic/depth_matterport_single_image',
                    batch_size=64, cleaned_data=False, crop=False, gray=False, rotate=False,
                    green=True, with_aug=True, depth_wo_body=False):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    if with_aug is False:
        transform = transforms.Compose([
            Scale([256, 256]),
            RandomHorizontalFlip(),
            CenterCrop([256, 256], [128, 128]),
            ToTensor(),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std'])
        ])
    else:
        transform = transforms.Compose([
            Scale([256, 256]),
            RandomHorizontalFlip(),
            CenterCrop([256, 256], [128, 128]),
            ToTensor(),
            Lighting(0.1, __imagenet_pca[
                'eigval'], __imagenet_pca['eigvec']),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std'])
        ])

    dataset = EgoDataset(data_dir=training_data_dir,
                         cleaned_data=cleaned_data,
                         transform=transform, crop=crop, gray=gray, green=green,
                         depth_wo_body=depth_wo_body)

    dataloader_training = DataLoader(dataset, batch_size,
                                     shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    return dataloader_training


def getTestingData(data_dir, batch_size=64, crop=False, total_length=1000, green=True):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    testing = EgoTestDataset(data_dir=data_dir,
                             transform=transforms.Compose([
                                 Scale([256, 256]),
                                 CenterCrop([256, 256], [128, 128]),
                                 ToTensor(),
                                 Normalize(__imagenet_stats['mean'],
                                           __imagenet_stats['std'])
                             ]),
                             total_length=total_length,
                             crop=crop, green=green)

    dataloader_testing = DataLoader(testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing


if __name__ == '__main__':
    dataset = EgoDataset(data_dir=r'\\winfs-inf\CT\EgoMocap\work\synthetic\matterport_with_seg_wo_body',
                         transform=transforms.Compose([
                             Scale([256, 256]),
                             RandomHorizontalFlip(),
                             # RandomRotate(5),
                             CenterCrop([256, 256], [128, 128]),
                             ToTensor(),
                             # Lighting(0.1, __imagenet_pca[
                             #     'eigval'], __imagenet_pca['eigvec']),
                             # ColorJitter(
                             #     brightness=0.4,
                             #     contrast=0.4,
                             #     saturation=0.4,
                             # ),
                             # Normalize(__imagenet_stats['mean'],
                             #           __imagenet_stats['std'])
                         ]), cleaned_data=False,
                         depth_wo_body=True)

    data = dataset[10000]
    img = data['image']
    depth = data['depth']
    # img.show()
    # depth.show()

    # depth = np.asarray(depth)
    print(depth.shape)
