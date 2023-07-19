import torch.nn.functional as F
import torch
import numpy as np

def get_rays(intrinsic, pose, img_size):
    '''
        intrinsics: [3, 3]
        pose: [3, 4]
        img_size: [H, W]
    '''
    origin = pose[:3, 3]
    rotm = pose[:3, :3]
    H, W = img_size[0], img_size[1]

    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    ray_dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    intrinsic_inv = np.linalg.inv(intrinsic)
    camera_dirs = ray_dirs @ intrinsic_inv.T
    directions = ((camera_dirs[Ellipsis, None, :] *
                rotm[None, None, :3, :3]).sum(axis=-1))
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    origins = np.tile(origin[None, None, :], (directions.shape[0], directions.shape[1], 1))

    return origins, directions

def get_rays_radii(intrinsic, pose, img_size):
    '''
    Add radii for Mip-NeRF
    Input:
        intrinsics: [3, 3]
        pose: [3, 4]
        img_size: [H, W]
    '''
    origin = pose[:3, 3]
    rotm = pose[:3, :3]
    H, W = img_size[0], img_size[1]

    focal = intrinsic[0, 0]

    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')

    ray_dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    intrinsic_inv = np.linalg.inv(intrinsic)
    camera_dirs = ray_dirs @ intrinsic_inv.T
    directions = ((camera_dirs[Ellipsis, None, :] *
                rotm[None, None, :3, :3]).sum(axis=-1))
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    origins = np.tile(origin[None, None, :], (directions.shape[0], directions.shape[1], 1))

    dx = np.sqrt(np.sum((directions[:-1, :, :] - directions[1:, :, :]) ** 2, -1))
    dx = np.concatenate([dx, dx[-2:-1, :]], 0)

    radii = dx[..., None] * 2 / np.sqrt(12)

    return origins, directions, radii

def sample_rays(rays_o, rays_d, rgb, num_samples):
    '''
        rays_o: [H, W, 3]
        rays_d: [H, W, 3]
        rgb: [H, W, 3]
        num_samples: int
    '''
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    rgb = rgb.view(-1, 3)

    sample_idx = np.random.choice(rays_o.shape[0], num_samples, replace=False)
    rays_o = rays_o[sample_idx]
    rays_d = rays_d[sample_idx]
    rgb = rgb[sample_idx]

    return rays_o, rays_d, rgb, sample_idx