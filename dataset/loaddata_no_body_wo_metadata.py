from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset.dataset_transforms import *
import os
import cv2
from tqdm import tqdm
from natsort import natsorted


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

    def __init__(self, data_dir_img, data_dir_depth,
                 cleaned_data, transform, crop=False, gray=False, green=True, depth_wo_body=False):
        self.data_dir_img = data_dir_img
        self.data_dir_depth = data_dir_depth
        self.transform = transform
        self.crop = crop
        self.gray = gray
        self.cleaned_data = cleaned_data
        self.green = green
        self.depth_wo_body = depth_wo_body

        self.data = []
        if self.cleaned_data is False:
            for scene_id in tqdm(os.listdir(self.data_dir_img)):
                scene_path_img = os.path.join(self.data_dir_img, scene_id)
                scene_path_depth = os.path.join(self.data_dir_depth, scene_id)
                if os.path.isdir(scene_path_img) is False or os.path.isdir(scene_path_depth) is False:
                    continue
                img_dir = os.path.join(scene_path_img, 'img')
                depth_dir = os.path.join(scene_path_depth, 'depth')
                for img_name in natsorted(os.listdir(img_dir)):
                    img_path = os.path.join(img_dir, img_name)
                    img_id = os.path.splitext(img_name)[0]
                    depth_path = os.path.join(depth_dir, img_id, 'Image0001.exr')
                    self.data.append({'img': img_path, 'depth': depth_path})
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
        if img is None:
            print('img read error at index: {}, img: {}'.format(index, self.data[index]['img']))
            return self.__getitem__((index + 1) % self.__len__())
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

        if self.gray:
            img = np.dot(img[..., :3], [0.299, 0.587, 0.144])
            img = np.stack([img, img, img], axis=-1)
            img = img.astype(np.uint8)

        # do a simple normalization
        depth = depth / 10. * 255.
        depth[depth > 255] = 255
        depth = depth.astype(np.uint8)

        img = Image.fromarray(img)
        depth = Image.fromarray(depth)
        sample = {'image': img, 'depth': depth}

        sample = self.transform(sample)

        return sample


def getTrainingData(training_data_dir_img=r'/CT/EgoMocap/static00/EgoGTAImages',
        training_data_dir_depth=r'/CT/EgoMocap/static00/EgoGTAImages_no_body',
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
    elif rotate:
        transform = transforms.Compose([
            Scale([256, 256]),
            RandomHorizontalFlip(),
            RandomRotate(5),
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

    print('using dataset: {}'.format(training_data_dir_img))
    dataset = EgoDataset(data_dir_img=training_data_dir_img,
                         data_dir_depth=training_data_dir_depth,
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
    dataset = EgoDataset(data_dir=r'\\winfs-inf\CT\EgoMocap\static00\EgoGTAImages_no_body',
                         transform=transforms.Compose([
                             Scale([256, 256]),
                             RandomHorizontalFlip(),
                             CenterCrop([256, 256], [128, 128]),
                         ]), cleaned_data=False)

    data = dataset[0]
    img = data['image']
    depth = data['depth']
    img.show()
