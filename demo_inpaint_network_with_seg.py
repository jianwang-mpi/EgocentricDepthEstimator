import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
from dataset.loaddata_demo_with_depth_seg import TestDatasetwithSeg
import os
import matplotlib
from tqdm import tqdm
from torch.utils.data import DataLoader

matplotlib.use('Agg')  # set the backend before importing pyplot
import matplotlib.image
import matplotlib.pyplot as plt
import cv2
import imageio
from models.unet import UNet
imageio.plugins.freeimage.download()
plt.set_cmap("jet")


def define_model(is_resnet, is_densenet, is_senet, deform=False):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], deformable=deform)
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208], deformable=deform)
    if is_senet:
        # original_model = senet.senet154(pretrained='imagenet')
        original_model = senet.senet154(pretrained=None)
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048], deformable=deform)

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')

    parser.add_argument('--img_dir', default=r'data/example_sequence',
                        type=str, help='name of the image directory')
    parser.add_argument('--model_path', default=r'logs/finetune_inpaint_network_with_seg/iter_2000.pth.tar',
                        type=str, help='name of the image directory')

    args = parser.parse_args()

    with torch.no_grad():
        model = define_model(is_resnet=False, is_densenet=False, is_senet=True, deform=False)
        saved_data = torch.load(args.model_path)
        model.load_state_dict(saved_data['state_dict'])
        model.cuda()
        model.eval()

        unet = UNet(n_channels=1, n_classes=1)
        unet.load_state_dict(saved_data['unet_state_dict'])
        unet.cuda()
        unet.eval()


        test_dataset = TestDatasetwithSeg(data_dir=args.img_dir, with_seg=True, seg_width=128)
        test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)
        for i, data_item in tqdm(enumerate(test_dataloader)):
            img_path = data_item['img_path'][0]
            image = data_item['img']
            seg = data_item['seg']
            image = image.cuda()
            seg = seg.cuda()
            # print(image[0])
            out = model(image)
            out = out * (1 - seg)
            out = unet(out)
            depth_dir = os.path.join(os.path.split(os.path.split(img_path)[0])[0], 'depths')
            if not os.path.isdir(depth_dir):
                os.mkdir(depth_dir)
            out_path = os.path.join(depth_dir, os.path.split(img_path)[1])
            out = out.view(out.size(2), out.size(3)).cpu().numpy()
            # print(out.shape)


            out = cv2.resize(out, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
            out = np.pad(out, ((0, 0), (64, 64)), 'constant', constant_values=0)

            # matplotlib.image.imsave(out_path, out)
            imageio.imwrite(out_path + '.exr', out, format='exr')


if __name__ == '__main__':
    main()
