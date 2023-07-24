import torch
import os
import numpy as np
from torch.utils.data import Dataset
import cv2

class WaymoDataset(Dataset):
    def __init__(self, config, phase='train'):
        super(WaymoDataset, self).__init__()

        self.config = config
        self.phase = phase
        self.scale_factor = config.scale_factor

        if self.phase == 'train':
            self.img_path = config.train_img_path
            self.poses_path = config.train_poses_path
        else:
            self.img_path = config.test_img_path
            self.poses_path = config.test_poses_path
        
        self.calib_path = config.calib_path
        self.cam_to_pose_path = config.cam_to_pose_path

        self.img_names = sorted(list(os.listdir(self.img_path)))

        self.load_imgs()
        self.read_intrinsics()
        self.read_poses()

        self.num_near_imgs = config.num_near_imgs
        self.near_img_idxs = self.get_near_img_idx()
        
        if self.config.depth_path is not None:
            self.read_depth()
        else:
            self.depths = None
        if self.config.normal_path is not None:
            self.read_normal()
        else:
            self.normals = None
    
    def load_imgs(self):
        imgs = []
        for img_name in self.img_names:
            img_path = os.path.join(self.img_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H,W,3
            if self.scale_factor != 1.0:
                w_s = int(img.shape[1] // self.scale_factor)
                h_s = int(img.shape[0] // self.scale_factor)
                img = cv2.resize(img, (w_s, h_s))
            img = img / 255.0
            imgs.append(img)
        
        imgs = np.stack(imgs, axis=0)
        self.imgs = imgs
    
    def read_poses(self):
        poses = np.load(self.poses_path)
        poses = poses[...,:3,:4]
        self.poses = poses
    
    def get_near_img_idx(self):
        all_poses = self.poses.copy()
        all_xyz = all_poses[:, :3, 3]

        dists = np.linalg.norm(all_xyz[:, None, :] - all_xyz[None, :, :], axis=-1)
        near_idx = np.argsort(dists, axis=-1)[:, 1:self.num_near_imgs+1]

        return near_idx
    
    def read_intrinsics(self):
        intrinsics = np.load(self.calib_path)

        if self.scale_factor != 1.0:
            intrinsics = intrinsics / self.scale_factor
            intrinsics[:,2,2] = 1.0
        self.intrinsics = intrinsics
    
    def read_depth(self):
        depths = []

        for img_name in self.img_names:
            depth_fn = img_name + '.geometric.npy'
            depth_fn = os.path.join(self.config.depth_path, depth_fn)
            depth = np.load(depth_fn)

            if self.scale_factor != 1.0:
                w_s = int(depth.shape[1] // self.scale_factor)
                h_s = int(depth.shape[0] // self.scale_factor)
                depth = cv2.resize(depth, (w_s, h_s))

            depths.append(depth)
        
        self.depths = np.stack(depths, axis=0)
        
    def read_normal(self):
        normals = []

        for img_name in self.img_names:
            normal_fn = img_name + '.geometric.npy'
            normal_fn = os.path.join(self.config.normal_path, normal_fn)
            normal = np.load(normal_fn)

            if self.scale_factor != 1.0:
                w_s = int(normal.shape[1] // self.scale_factor)
                h_s = int(normal.shape[0] // self.scale_factor)
                normal = cv2.resize(normal, (w_s, h_s))

            normals.append(normal)
        
        self.normals = np.stack(normals, axis=0)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        img = torch.from_numpy(img).float() # [H,W,3]
        data_dict = {}
        data_dict['rgb'] = img
        data_dict['pose'] = torch.from_numpy(self.poses[index]).float()
        data_dict['intrinsic'] = torch.from_numpy(self.intrinsics).float()
        data_dict['img_hw'] = (torch.tensor([img.shape[0], img.shape[1]])).float()
        data_dict['img_name'] = self.img_names[index]

        if self.depths is not None:
            data_dict['depth'] = torch.from_numpy(self.depths[index]).float()
        if self.normals is not None:
            data_dict['normal'] = torch.from_numpy(self.normals[index]).float()
        
        return data_dict
    
    def __len__(self):
        return self.imgs.shape[0]