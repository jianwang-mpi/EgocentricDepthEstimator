# Egocentric Depth Estimator

This is the repo for work on egocentric depth estimation.

This is a part of the paper:

**Scene-aware Egocentric 3D Human Pose Estimation**

*Jian Wang, Diogo Luvizon, Weipeng Xu, Lingjie Liu, Kripasindhu Sarkar, Christian Theobalt*

*CVPR 2023*

[[Project Page](https://vcai.mpi-inf.mpg.de/projects/sceneego/)]  [[Datasets](https://nextcloud.mpi-klsb.mpg.de/index.php/s/Ritzm3ycioAADSH)]

### Run the demo

1. Download the [pretrained model](https://nextcloud.mpi-klsb.mpg.de/index.php/s/cWqcdk2KxKZnbim) and put it in the `checkpoints` folder.
2. Run the demo script:

```bash
python demo.py --img_dir ./data/example_sequence --model_path ./checkpoints/ego_depth.pth.tar
```
3. The estimated depth maps will be saved in the `data/example_sequence/depths` folder.
