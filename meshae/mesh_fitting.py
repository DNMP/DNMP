import torch
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
import os
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import point_mesh_face_distance, mesh_laplacian_smoothing, chamfer_distance, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm
import sys
import argparse

# Basic args
recenter = True
batch_size = 32
device = 'cuda:0'
num_iter = 500

data_dir = '' # extracted point cloud patches
out_dir = ''

fns = sorted(os.listdir(data_dir))

pts_list = []
for fn in tqdm(fns):
    pts_path = os.path.join(data_dir, fn)
    pts = np.load(pts_path)
    pts_list.append(pts)
pts_list = np.array(pts_list)

if recenter:
    pts_mean = torch.mean(pts_list, dim=1, keepdim=True) # [M, 1, 3]
    pts_list = pts_list - pts_mean
    pts_max_dim = torch.max(torch.abs(pts_list), dim=1, keepdim=True)[0] # [M, 1, 3]
    pts_max_dim = torch.max(pts_max_dim, dim=2, keepdim=True)[0] # [M, 1, 1]
    pts_list = pts_list / (pts_max_dim + 1e-8)

for idx in tqdm(range(pts_list.shape[0]//batch_size)):
    points = pts_list[idx*batch_size:(idx+1)*batch_size]
    batch_fns = fns[idx*batch_size:(idx+1)*batch_size]
    points = points.to(device)
    temp_mesh = ico_sphere(1, device=device)
    verts = temp_mesh.verts_packed()
    faces = temp_mesh.faces_packed()
    verts = verts.unsqueeze(0).repeat(batch_size, 1, 1)
    faces = faces.unsqueeze(0).repeat(batch_size, 1, 1)
    src_mesh = Meshes(verts, faces).to(device)
    verts_shape = src_mesh.verts_packed().shape
    offset = torch.zeros(verts_shape, requires_grad=True, device=device)
    pc = Pointclouds(points)
    optimizer = torch.optim.SGD([offset], lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250, 400], gamma=0.1)
    for i in range(num_iter):
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(offset)
        num_samples = points.shape[1]
        sampled_points = sample_points_from_meshes(new_src_mesh, num_samples=num_samples)
        chamfer_loss, _ = chamfer_distance(sampled_points, points)
        smooth_loss = mesh_laplacian_smoothing(new_src_mesh)
        normal_loss = mesh_normal_consistency(new_src_mesh)
        loss = chamfer_loss
        loss = loss + smooth_loss * 0.05 + normal_loss * 0.05
        loss = loss * batch_size
        if i % 100 == 0:
            loss_ = loss.item() / batch_size
            print(f'{idx}: {i}: {loss_}')
            sys.stdout.flush()
        loss.backward()
        optimizer.step()
        scheduler.step()

    new_src_mesh = src_mesh.offset_verts(offset)
    verts_optimized = new_src_mesh.verts_packed().view(-1, 42, 3)
    faces_optimized = faces
    verts_optimized = verts_optimized.detach().cpu().numpy()
    faces_optimized = faces_optimized.cpu().numpy()
    for local_idx in range(batch_size):
        fn = batch_fns[local_idx]
        if fn.endswith('.npy'):
            new_fn = fn.replace('.npy', '.npz')
        else:
            new_fn = fn
        fn = os.path.join(out_dir, new_fn)
        verts_ = verts_optimized[local_idx]
        faces_ = faces_optimized[local_idx]
        np.savez(fn, verts=verts_, faces=faces_)