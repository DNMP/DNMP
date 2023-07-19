from io import BytesIO
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import cv2

class KITTI360Dataset(Dataset):
    def __init__(self, config, phase='train'):
        super(KITTI360Dataset, self).__init__()

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

        self.img_names = sorted(os.listdir(self.img_path))

        self.load_imgs()
        self.read_intrinsics()
        self.read_cam_to_pose()
        self.read_poses()

        self.num_near_imgs = config.num_near_imgs
        self.near_img_idxs = self.get_near_img_idx()
        
        if self.config.depth_path is not None and self.phase == 'train':
            self.read_depth()
        else:
            self.depths = None
        if self.config.normal_path is not None and self.phase == 'train':
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
        
        self.H, self.W = imgs[0].shape[0], imgs[0].shape[1]
        self.imgs = imgs
    
    def read_poses(self):

        img_pose_dict = {}
        
        with open(self.poses_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').strip(' ').split(' ')
                img_id = int(line[0])
                pose = np.array(line[1:], dtype=np.float32)
                pose = pose.reshape(3, 4)
                pose = pose[:3, :4]
                img_pose_dict[img_id] = pose

        poses = []
        for img_name in self.img_names:
            cam_id = img_name.split('.')[0].split('_')[0]
            if cam_id == '00':
                c2p = self.c2p_00
                R_rect = self.R_rect_00
            elif cam_id == '01':
                c2p = self.c2p_01
                R_rect = self.R_rect_01

            img_id = int(img_name.split('.')[0].split('_')[-1])
            pose = img_pose_dict[img_id]
            pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
            c2w = np.matmul(np.matmul(pose, c2p), np.linalg.inv(R_rect))

            poses.append(c2w[:3,:4])

        poses = np.stack(poses, axis=0)
        self.poses = poses
    
    def get_near_img_idx(self):
        all_poses = self.poses.copy()
        all_xyz = all_poses[:, :3, 3]

        dists = np.linalg.norm(all_xyz[:, None, :] - all_xyz[None, :, :], axis=-1)
        near_idx = np.argsort(dists, axis=-1)[:, 1:self.num_near_imgs+1]

        return near_idx
    
    def read_intrinsics(self):

        with open(self.calib_path, 'r') as f:
            lines = f.readlines()

        P0 = lines[9].strip('\n').split(' ')
        assert P0[0] == 'P_rect_00:'
        P0 = np.array(P0[1:], dtype=np.float32)
        P0 = P0.reshape(3, 4)
        intrinsics = P0[:3, :3]
        if self.scale_factor != 1.0:
            intrinsics = intrinsics / self.scale_factor
            intrinsics[2,2] = 1.0
        self.intrinsics = intrinsics

        R_rect_00 = lines[8].strip('\n').split(' ')
        assert R_rect_00[0] == 'R_rect_00:'
        R_rect_00 = np.array(R_rect_00[1:], dtype=np.float32)
        R_rect_00_ = R_rect_00.reshape(3, 3)
        R_rect_00 = np.eye(4)
        R_rect_00[:3, :3] = R_rect_00_
        self.R_rect_00 = R_rect_00

        R_rect_01 = lines[16].strip('\n').split(' ')
        assert R_rect_01[0] == 'R_rect_01:'
        R_rect_01 = np.array(R_rect_01[1:], dtype=np.float32)
        R_rect_01_ = R_rect_01.reshape(3, 3)
        R_rect_01 = np.eye(4)
        R_rect_01[:3, :3] = R_rect_01_
        self.R_rect_01 = R_rect_01
    
    def read_cam_to_pose(self):

        with open(self.cam_to_pose_path, 'r') as f:
            lines = f.readlines()
       
        lines = [line.strip() for line in lines]
        for line in lines:
            line = line.split(' ')
            if line[0] == 'image_00:':
                c2p_00 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
            elif line[0] == 'image_01:':
                c2p_01 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
        
        c2p_00 = np.concatenate([c2p_00, np.array([[0, 0, 0, 1]])], axis=0)
        c2p_01 = np.concatenate([c2p_01, np.array([[0, 0, 0, 1]])], axis=0)

        self.c2p_00 = c2p_00
        self.c2p_01 = c2p_01
    
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
        data_dict['img_hw'] = (torch.tensor([self.H, self.W])).float()
        data_dict['img_name'] = self.img_names[index]

        near_img_idx = self.near_img_idxs[index]
        near_imgs = self.imgs[near_img_idx]
        near_poses = self.poses[near_img_idx]
        data_dict['near_imgs'] = torch.from_numpy(near_imgs).float()
        data_dict['near_poses'] = torch.from_numpy(near_poses).float()

        if self.depths is not None:
            data_dict['depth'] = torch.from_numpy(self.depths[index]).float()
            near_depths = torch.from_numpy(self.depths[near_img_idx]).float()
            data_dict['near_depths'] = near_depths
        if self.normals is not None:
            data_dict['normal'] = torch.from_numpy(self.normals[index]).float()
            near_normals = torch.from_numpy(self.normals[near_img_idx]).float()
            data_dict['near_normals'] = near_normals
        
        return data_dict
    
    def __len__(self):
        return self.imgs.shape[0]