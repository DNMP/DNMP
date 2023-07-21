# [ICCV 2023] Urban Radiance Field Representation with Deformable Neural Mesh Primitives

## Introduction
The repository contains the source code and pre-trained models of our paper (published on ICCV 2023): Urban Radiance Field Representation with Deformable Neural Mesh Primitives.


## Updates
- 2023.07.21: The :fire::fire::fire:**The [pre-print](https://arxiv.org/abs/2307.10776) is released!**:fire::fire::fire: Refer to it for more details!
- 2023.07.19: The [project page](https://dnmp.github.io/) is created. Check it out for an overview of our work!

## Datasets
We conduct experiments on two outdoor datasets: KITTI-360 dataset, Waymo-Open-Dataset.
Please refer to preprocess/README.md for more details.

## Environments

1. Compile fairnr.
```
python setup.py build_ext --inplace
```

2. Main requirements:
- CUDA (tested on cuda-11.1)
- PyTorch (tested on torch-1.9.1)
- [pytorch3d](https://pytorch3d.org/)
- [torchsparse](https://github.com/mit-han-lab/torchsparse)

3. Other requirements are provided in `requirements.txt`

## Training
1. Optimize geometry using our pre-trained auto-encoder by running `sh scripts/train_${DATASET}_geo.sh`. (Please specify `SEQUENCE`,`DATA_ROOT`,`LOG_DIR` and `CKPT_DIR` in the script.)

2. Train radiance field by running `sh scripts/train_${DATASET}_render.sh`. (Please specify `SEQUENCE`,`DATA_ROOT`,`LOG_DIR`, `CKPT_DIR` and `PRETRAINED_GEO` in the script.)

## Evaluation

You can run `scripts/test_kitti360.sh` for evaluation. (Please specify `SAVE_DIR`, `DATA_ROOT` and the pretrained files in the script.)

## To-Do List

- [ ] Release Code and pretrained model
- [x] Technical Report
- [x] Project page

## Citation
If you find this project useful for your work, please consider citing:
```
@article{lu2023dnmp,
  author    = {Lu, Fan and Xu, Yan and Chen, Guang and Li, Hongsheng and Lin, Kwan-Yee and Jiang, Changjun},
  title     = {Urban Radiance Field Representation with Deformable Neural Mesh Primitives},
  journal   = {ICCV},
  year      = {2023},
}
```
