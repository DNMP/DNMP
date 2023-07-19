import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from datasets import dataset_dict
from models.renderer import DNMPGeometry
from models.mesh_AE import PointDecoder
from models.ray_utils import get_rays, sample_rays

from core.cameras import get_camera
from core.train_utils import AverageMeter, ensure_dir, get_logger
import core.losses as losses

from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
import json
import shutil

class TrainerGeometry:
    def __init__(self, config):
        self.config = config
        
        ensure_dir(self.config.log_dir)
        self.logger = get_logger(os.path.join(self.config.log_dir, 'train.log'))
        self.logger.info(self.config)
        self.writer = SummaryWriter(log_dir=os.path.join(self.config.log_dir, 'tensorboard'))
        self.checkpoint_dir = self.config.checkpoint_dir
        ensure_dir(self.checkpoint_dir)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            raise ValueError('GPU not available')

        # Recenter the scene using center point
        if self.config.center_point_fn is not None:
            center_point = np.load(self.config.center_point_fn)
            self.transform_M = np.linalg.inv(center_point)
        else:
            self.transform_M = np.eye(4)
        
        # Load and recenter point cloud
        pts_data = np.load(self.config.pts_file)
        pts = pts_data['pts']
        pts = np.matmul(self.transform_M, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).transpose()).transpose()[:, :3]

        # Load pretrained mesh auto-encoder
        assert self.config.pretrained_mesh_ae is not None
        self.decoder_fn = PointDecoder(config.mesh_ae_hidden_size, voxel_size=1.).to(self.device)
        decoder_dict = torch.load(self.config.pretrained_mesh_ae)
        self.decoder_fn.load_state_dict(decoder_dict['decoder_state_dict'])
        self.decoder_fn.requires_grad = True

        self.render = DNMPGeometry(self.config, self.device, pts)

        config_dict = vars(self.config)
        config_file = os.path.join(self.config.log_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=1)
        
        snapshot_dir = os.path.join(self.config.log_dir, 'snapshot')
        ensure_dir(snapshot_dir)

        BASE_DIR = os.getcwd()
        
        shutil.copytree(os.path.join(BASE_DIR, 'models'), \
            os.path.join(os.path.join(snapshot_dir, 'models')))
        shutil.copytree(os.path.join(BASE_DIR, 'core'), \
            os.path.join(os.path.join(snapshot_dir, 'core')))
        shutil.copytree(os.path.join(BASE_DIR, 'scripts'), \
            os.path.join(os.path.join(snapshot_dir, 'scripts')))

        self.optimizer = Adam([{'params': self.render.parameters()}], 
                               lr=self.config.lr,
                               betas=(0.9, 0.999),
                               weight_decay=0.)
        
        self.scheduler = ExponentialLR(self.optimizer, 0.999999)
        self.global_step = 0
        self.val_step = 0

        self.max_iter = self.config.max_iter
        self.best_val_psnr = 0.0

        self.get_dataloader()

    def get_dataloader(self):
        dataset = dataset_dict[self.config.dataset]

        self.train_dataset = dataset(self.config, phase='train')
        self.val_dataset = dataset(self.config, phase='val')

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=1,
                                                   shuffle=True,
                                                   num_workers=self.config.num_workers,
                                                   pin_memory=False)

        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.config.num_workers,
                                                 pin_memory=False)
        
        self.val_iter = self.val_loader.__iter__()
        self.val_iter_num = 0
        self.max_val_iter_num = len(self.val_loader)
    
    def train(self):
        self.render.train()
        loss_meter = AverageMeter()

        flag = True
        while flag:
            for idx, sample in enumerate(self.train_loader):

                if self.global_step > self.max_iter:
                    flag = False
                    break
                self.global_step += 1
                self.optimizer.zero_grad()
                rgb = sample['rgb'].squeeze()
                pose = sample['pose'].squeeze().numpy()
                depth = sample['depth'].squeeze()
                intrinsic = sample['intrinsic'].squeeze().numpy()
                img_hw = sample['img_hw'].squeeze().numpy()
                
                pose_ = np.eye(4)
                pose_[:3, :4] = pose
                pose = np.matmul(self.transform_M, pose_)

                if self.config.use_photo_check or self.config.use_photo_loss:
                    near_imgs = sample['near_imgs'].squeeze()
                    near_poses = sample['near_poses'].squeeze().numpy()
                    near_poses_ = np.eye(4)[None,...].repeat(near_poses.shape[0], axis=0)
                    near_poses_[:, :3, :4] = near_poses
                    near_poses = np.matmul(self.transform_M[None,...], near_poses_)
                    near_poses = near_poses[:,:3,:4]

                if 'normal' in sample.keys() and self.config.use_normal_loss:
                    normal = sample['normal'].squeeze()
                    # normal from COLMAP need to be transformed to world coordinate
                    c2w_rot = torch.from_numpy(pose[:3,:3]).to(self.device).float()
                    normal = torch.matmul(c2w_rot, normal.permute(1,0)).permute(1,0)
                    transform_rot = torch.from_numpy(self.transform_M[:3,:3]).to(self.device).float()
                    normal = (torch.matmul(transform_rot, normal.permute(1,0))).permute(1,0)
                
                rays_o, rays_d = get_rays(intrinsic, pose, img_hw)
                rays_o = torch.from_numpy(rays_o).float().to(self.device)
                rays_d = torch.from_numpy(rays_d).float().to(self.device)
                rgb = rgb.to(self.device)
                depth = depth.to(self.device)

                camera = get_camera(intrinsic, pose, img_hw, self.device)
                rays_o_, rays_d_, rgb_, pix_coords_idx = sample_rays(rays_o, rays_d, rgb, self.config.num_rays)
                pix_coords_idx = torch.from_numpy(pix_coords_idx).long().to(self.device)

                ret_dict = self.render.render_depth_normal(rays_o_, rays_d_, camera, pix_coords_idx, self.decoder_fn, img_hw)

                depth_ = depth.view(-1)[pix_coords_idx]
                depth_loss = (torch.abs(ret_dict['depth'] - depth_))

                valid_idx_mesh = (ret_dict['depth'] > 0)
                valid_idx_gt = (depth_ > 0) & (depth_ < self.config.valid_depth_thresh)

                depth_valid_idx = valid_idx_mesh & valid_idx_gt
                
                if self.config.use_photo_check:
                    gt_depth_check = losses.photometric_loss(
                        pred_rgb=rgb_,
                        pix_coords_idx=pix_coords_idx,
                        ref_rgbs=near_imgs.to(self.device),
                        intrinsic=torch.from_numpy(intrinsic).float().to(self.device),
                        pred_depth=depth_,
                        pred_pose=torch.from_numpy(pose).float().to(self.device),
                        ref_poses=torch.from_numpy(near_poses).float().to(self.device),
                        reduction='minimum'
                    )
                    
                    valid_idx_photo = gt_depth_check < self.config.photo_check_thresh
                    depth_valid_idx = depth_valid_idx & valid_idx_photo
                
                depth_loss = depth_loss[depth_valid_idx]
                if depth_loss.shape[0] == 0:
                    self.logger.info('Get invalid depth loss, skip this batch')
                    continue
                else:
                    depth_loss = depth_loss.mean()
                
                loss = depth_loss * self.config.depth_loss_weight
                
                if self.config.use_normal_loss:
                    normal_ = normal[pix_coords_idx]
                    normal_loss = ((torch.abs(ret_dict['normal'] - normal_)).sum(-1))
                    normal_norm = torch.norm(ret_dict['normal'], dim=-1)
                    valid_idx_norm = (normal_norm > 0)
                    normal_valid_idx = valid_idx_norm & depth_valid_idx
                    normal_loss = normal_loss[normal_valid_idx]
                    loss = loss + normal_loss * self.config.normal_loss_weight
                
                if self.config.use_photo_loss:
                    photo_loss = losses.photometric_loss_multi(
                        pred_rgb=rgb_,
                        pix_coords_idx=pix_coords_idx,
                        ref_rgbs=near_imgs.to(self.device),
                        intrinsic=torch.from_numpy(intrinsic).float().to(self.device),
                        pred_depth=ret_dict['depth'],
                        pred_pose=torch.from_numpy(pose).float().to(self.device),
                        ref_poses=torch.from_numpy(near_poses).float().to(self.device),
                        reduction='mean'
                    )
                    loss = loss + photo_loss * self.config.photo_loss_weight
                
                if self.config.use_deform_loss:
                    loss = loss + ret_dict['deform_loss'] * self.config.deform_loss_weight
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_meter.update(loss.item())

                if self.global_step % self.config.print_freq == 0:
                    self.logger.info(
                        'Iter [{}/{}] Loss: {:.4f}'.format(
                            self.global_step, self.max_iter, loss_meter.avg))

                    self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                    loss_meter.reset()
                
                if self.global_step % self.config.val_freq == 0:
                    self.validate()
                    self.save_ckpt(f'ckpt_{self.global_step}')
                    self.logger.info(f'Checkpoint saved at step {self.global_step}')
    
    def validate(self):

        self.render.config.perturb = 0.0

        with torch.no_grad():
            # only evaluate one image to save time
            sample = self.val_iter.next()
            self.val_iter_num += 1
            if self.val_iter_num == self.max_val_iter_num:
                self.val_iter = self.val_loader.__iter__()
                self.val_iter_num = 0

            rgb = sample['rgb'].to(self.device).squeeze()
            pose = sample['pose'].squeeze().numpy()
            intrinsic = sample['intrinsic'].squeeze().numpy()
            img_hw = sample['img_hw'].squeeze().numpy()
            pose_ = np.eye(4)
            pose_[:3, :4] = pose
            pose = np.matmul(self.transform_M, pose_)
            rays_o, rays_d = get_rays(intrinsic, pose, img_hw)

            rays_o = torch.from_numpy(rays_o).float().to(self.device)
            rays_d = torch.from_numpy(rays_d).float().to(self.device)

            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)

            camera = get_camera(intrinsic, pose, img_hw, self.device)
            pix_coords_idx = torch.arange(0, rays_o.shape[0]).long().to(self.device)

            ret_dict = self.render.render_depth_normal(rays_o, rays_d, camera, pix_coords_idx, self.decoder_fn, img_hw)

            rgb = rgb.cpu().numpy()

            pred_depth = ret_dict['depth'].view(int(img_hw[0]), int(img_hw[1]))
            pred_depth = pred_depth.cpu().numpy()
            pred_depth = pred_depth / (np.max(pred_depth) + 1e-8)

        self.writer.add_image('val/gt_image', rgb, self.val_step, dataformats='HWC')
        self.writer.add_image('val/pred_depth', pred_depth, self.val_step, dataformats='HW')

        self.val_step += 1
    
    def save_ckpt(self, filename='checkpoint'):

        self.logger.info(f'Save checkpoint for iteration {self.global_step}.')
        state = {
            'global_step': self.global_step,
            'state_dict': self.render.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
        }
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        torch.save(state, filename)
