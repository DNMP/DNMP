import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset

class MeshDataset(Dataset):

    def __init__(self, config, phase='train', augmentation=True):
        super(MeshDatasetAWS, self).__init__()

        self.mesh_dir = config.mesh_dir
        self.augmentation = augmentation

        mesh_fns = sorted(os.listdir(self.mesh_dir))
        num_fns = len(mesh_fns)
        self.split_dataset(num_fns, phase)
        self.mesh_fns = [mesh_fns[i] for i in self.sample_idx]
        self.vertices_list, self.faces_list = self.load_meshes()
    
    def split_dataset(self, num_pts, phase):
        all_idx = np.arange(0, num_pts)
        val_idx = np.arange(0, num_pts, 10)
        train_idx = np.delete(all_idx, val_idx)
        if phase == 'train':
            self.sample_idx = train_idx
        elif phase == 'val':
            self.sample_idx = val_idx
    
    def load_meshes(self):
        vertices_list = []
        faces_list = []
        print('loading meshes....')
        for mesh_fn in tqdm(self.mesh_fns):
            mesh_path = os.path.join(self.mesh_dir, mesh_fn)
            data = np.load(mesh_path)
            vertices = data['vertices']
            faces = data['faces']
            vertices_list.append(vertices)
            faces_list.append(faces)
        return vertices_list, faces_list
    
    def generate_random_pose(self, max_rotation=5.0, max_translation=0.1):
        random_rot = np.random.uniform(low=-max_rotation, high=max_rotation, size=(3,)) * np.pi / 180.
        random_trans = np.random.uniform(low=-max_translation, high=max_translation, size=(3,))
        rot1 = np.array(
            [[1., 0., 0.],
            [0., np.cos(random_rot[0]), -np.sin(random_rot[0])],
            [0., np.sin(random_rot[0]), np.cos(random_rot[0])]]
        )
        rot2 = np.array(
            [[np.cos(random_rot[1]), 0., np.sin(random_rot[1])],
            [0., 1., 0.],
            [-np.sin(random_rot[1]), 0., np.cos(random_rot[1])]]
        )
        rot3 = np.array(
            [[np.cos(random_rot[2]), -np.sin(random_rot[2]), 0.],
            [np.sin(random_rot[2]), np.cos(random_rot[2]), 0.],
            [0., 0., 1.]]
        )
        rotm = rot3.dot(rot2.dot(rot1))
        random_pose = np.eye(4)
        random_pose[:3,:3] = rotm
        random_pose[:3,3] = random_trans

        return random_pose
    
    def __getitem__(self, index):
        
        vertices = self.vertices_list[index]
        faces = self.faces_list[index]
        if self.augmentation:
            random_pose = self.generate_random_pose(max_rotation=180, max_translation=0.0)
            vertices = vertices.dot(random_pose[:3,:3].T) + random_pose[:3,3][None,:]
        
        sample = {}

        vertices = torch.from_numpy(vertices).float()
        faces = torch.from_numpy(faces).long()
        sample['vertices'] = vertices
        sample['faces'] = faces

        return sample
    
    def __len__(self):
        return len(self.vertices_list)