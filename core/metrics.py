import torch
import numpy as np
from skimage.metrics import structural_similarity
import torch.nn.functional as F
import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg')

mse2psnr = lambda x:-10. * torch.log(x) / (torch.log(torch.tensor([10.0])).item())

def mse2psnr_np(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / np.log(10.) * np.log(mse)

def ssim_fn(x, y):
    return structural_similarity(x, y, multichannel=True)

def lpips_fn(x, y):
    '''
        x: [H,W,3]
        y: [H,W,3]
    '''
    x = torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0)
    y = torch.from_numpy(y).float().permute(2,0,1).unsqueeze(0)

    loss = loss_fn_vgg(x, y, normalize=True)
    loss = loss.item()
    return loss

def check_depth_loss(pred_rgb, pix_coords_idx, ref_rgbs, intrinsic, pred_depth, pred_pose, ref_poses, reduction='mean'):
    '''
        pred_rgb: [N,3]
        pix_coords_idx: [N,]
        ref_rgb: [H,W,3]
        intrinsic: [3,3]
        pred_depth: [N]
        pred_pose: [3,4] (cam2world)
        ref_pose: [V,3,4] (cam2world)
        reduction: 'mean' or 'minimum
    '''
    mask_depth = (pred_depth > 0)
    def inbound(pixel_locations, h, w):
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def photometric_loss_single(pred_rgb, pix_coords_idx, ref_rgb, intrinsic, pred_depth, pred_pose, ref_pose):
        '''
            pred_rgb: [N,3]
            pix_coords_idx: [N,]
            ref_rgb: [H,W,3]
            intrinsic: [3,3]
            pred_depth: [N]
            pred_pose: [3,4] (cam2world)
            ref_pose: [3,4] (cam2world)
        '''
        num_pts = pix_coords_idx.shape[0]
        H, W = ref_rgb.shape[0], ref_rgb.shape[1]
        x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
        coords = np.stack([x, y], axis=-1)
        coords = torch.from_numpy(coords).float().to(pred_rgb.device)
        coords = coords.view(-1,2)
        pred_coords = coords[pix_coords_idx]

        inv_K = (torch.inverse(intrinsic.unsqueeze(0))).repeat(num_pts, 1, 1) # [N,3,3]
        pred_coords_h = torch.cat([pred_coords, torch.ones_like(pred_coords[...,:1])], dim=-1) # [N,3]
        pred_depth = pred_depth[:,None,None] # [N,1,1]
        pred_xyz_cam = torch.matmul(inv_K, pred_coords_h.unsqueeze(-1)) * pred_depth # [N,3,1]

        pred_xyz_cam_h = torch.cat([pred_xyz_cam, torch.ones_like(pred_xyz_cam[:,:1,:])], dim=1) # [N,4,1]

        pred_pose_ = torch.eye(4).to(pred_pose)
        pred_pose_[:3,:4] = pred_pose[:3,:4]
        pred_pose_ = pred_pose_.unsqueeze(0).repeat(num_pts, 1, 1) # [N,4,4]
        pred_xyz_world = (torch.matmul(pred_pose_, pred_xyz_cam_h)) # [N,4,1]

        ref_pose_ = torch.eye(4).to(ref_pose)
        ref_pose_[:3,:4] = ref_pose[:3,:4]
        ref_pose_inv = (torch.inverse(ref_pose_.unsqueeze(0))).repeat(num_pts, 1, 1)
        ref_xyz_cam_h = torch.matmul(ref_pose_inv, pred_xyz_world) # [N,4,1]
        ref_xyz_cam = ref_xyz_cam_h[:,:3,:] # [N,3,1]

        ref_coords = torch.matmul(intrinsic, ref_xyz_cam) # [N,3,1]
        ref_coords = ref_coords[:,:2,0] / (ref_coords[:,2:,0]+1e-4) # [N,2]
        mask = inbound(ref_coords, H, W) # [N]

        xy_cam = ref_coords[None,:,:] # [1,N,2]
        xy_cam_patch = xy_cam.unsqueeze(2).repeat(1,1,9,1)
        box_offset = torch.tensor(
            [[0.,0.],
             [1.,-1.],
             [1.,1.],
             [1.,0.],
             [-1.,0.],
             [-1.,1.],
             [-1.,-1.],
             [0.,-1.],
             [0.,1.]]).to(xy_cam)
        box_offset = box_offset[None,None,:,:] # [1,1,9,2]
        xy_cam_patch = xy_cam_patch + box_offset
        xy_cam_patch = xy_cam_patch.view(1,-1,2)

        resize_factor = torch.tensor([W-1., H-1.]).to(xy_cam.device)[None, None, :]
        normalized_pixel_locations = 2 * xy_cam_patch / resize_factor - 1.  # [1, N, 2]
        normalized_pixel_locations = normalized_pixel_locations.unsqueeze(2) # [1, N, 1, 2]
        ref_rgb = ref_rgb.unsqueeze(0).permute(0,3,1,2) # [1, H, W, 3]
        rgb_sampled = F.grid_sample(ref_rgb, normalized_pixel_locations, align_corners=True) # [1, C, N, 1]
        rgb_sampled = rgb_sampled.squeeze() # [3, N]
        rgb_sampled = rgb_sampled.permute(1,0) # [N, 3]
        rgb_sampled = rgb_sampled.view(-1,9,3)
        rgb_sampled = torch.min(rgb_sampled, dim=1)[0]

        loss = (pred_rgb - rgb_sampled)**2

        mask = mask & mask_depth

        return loss, mask
    
    losses = []
    masks = []
    for idx in range(ref_rgbs.shape[0]):
        ref_rgb = ref_rgbs[idx]
        ref_pose = ref_poses[idx]

        loss, mask = photometric_loss_single(pred_rgb, pix_coords_idx, ref_rgb, intrinsic, pred_depth, pred_pose, ref_pose)
        losses.append(loss)
        masks.append(mask)
    
    losses = torch.stack(losses, dim=0) # [V, N, 3]
    losses = torch.mean(losses, dim=-1, keepdim=False) # [V, N]
    masks = torch.stack(masks, dim=0) # [V, N]

    losses[~masks] = 1e10
    min_losses, min_idx = torch.min(losses, dim=0, keepdim=False) # [N]
    
    return min_losses