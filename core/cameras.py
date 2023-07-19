import torch
import pytorch3d
import numpy as np
from pytorch3d.renderer.cameras import PerspectiveCameras

def get_camera(intrinsic, pose, hw, device):
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    num_cameras = 1
    focal_length = np.zeros((num_cameras, 2))
    focal_length[:, 0] = fx
    focal_length[:, 1] = fy
    principal_point = np.zeros((num_cameras, 2))
    principal_point[:, 0] = cx
    principal_point[:, 1] = cy
    focal_length = torch.from_numpy(focal_length).float()
    principal_point = torch.from_numpy(principal_point).float()
    w = hw[1]
    h = hw[0]
    half_imwidth = w/2.0
    half_imheight = h/2.0
    focal_length[:,0] /= half_imheight
    focal_length[:,1] /= half_imheight

    principal_point[:, 0] = -(principal_point[:, 0]-half_imwidth)/half_imheight
    principal_point[:, 1] = -(principal_point[:, 1]-half_imheight)/half_imheight
    mirror = torch.eye(4).unsqueeze(0).to(device).repeat(num_cameras, 1, 1)
    mirror[:, 0, 0] = -1.0
    mirror[:, 1, 1] = -1.0

    R_T_joined = torch.eye(4).unsqueeze(0).repeat(num_cameras,1,1).to(device)
    if not torch.is_tensor(pose):
        pose = torch.from_numpy(pose).float()
    pose = pose.to(device)
    R = pose[:3,:3].to(device)
    T = pose[:3,3].to(device)
    R_T_joined[:, :3, :3] = R
    R_T_joined[:, :3, 3]  = T
    R_T_joined = torch.inverse(R_T_joined)
    new_R_T = torch.bmm(mirror, R_T_joined)
    R_camera = new_R_T[:, :3, :3]
    T_camera = new_R_T[:, :3,  3]
    cameras = PerspectiveCameras(
        device=device, 
        R=R_camera.transpose(1,2), 
        T=T_camera, focal_length=focal_length, 
        principal_point= principal_point)
    return cameras

def get_camera_no_ndc(intrinsic, pose, hw, device):
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    num_cameras = 1
    focal_length = np.zeros((num_cameras, 2))
    focal_length[:, 0] = fx
    focal_length[:, 1] = fy
    principal_point = np.zeros((num_cameras, 2))
    principal_point[:, 0] = cx
    principal_point[:, 1] = cy
    focal_length = torch.from_numpy(focal_length).float()
    principal_point = torch.from_numpy(principal_point).float()
    w = hw[1]
    h = hw[0]
    half_imwidth = w/2.0
    half_imheight = h/2.0
    img_size = torch.tensor([h, w]).float().to(device).unsqueeze(0)

    mirror = torch.eye(4).unsqueeze(0).to(device).repeat(num_cameras, 1, 1)
    mirror[:, 0, 0] = -1.0
    mirror[:, 1, 1] = -1.0

    R_T_joined = torch.eye(4).unsqueeze(0).repeat(num_cameras,1,1).to(device)
    if not torch.is_tensor(pose):
        pose = torch.from_numpy(pose).float()
    pose = pose.to(device)
    R = pose[:3,:3].to(device)
    T = pose[:3,3].to(device)
    R_T_joined[:, :3, :3] = R
    R_T_joined[:, :3, 3]  = T
    R_T_joined = torch.inverse(R_T_joined)
    new_R_T = torch.bmm(mirror, R_T_joined)
    R_camera = new_R_T[:, :3, :3]
    T_camera = new_R_T[:, :3,  3]
    cameras = PerspectiveCameras(
        device=device, 
        R=R_camera.transpose(1,2), 
        T=T_camera, 
        focal_length=focal_length, 
        principal_point=principal_point, 
        in_ndc=False, image_size=img_size)
    return cameras