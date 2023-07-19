import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from datasets import dataset_dict

from models.renderer import DNMPRender
from models.mesh_AE import PointDecoder
from models.ray_utils import get_rays, sample_rays

from core.cameras import get_camera
from core.train_utils import AverageMeter, ensure_dir, get_logger
import core.metrics as metrics

from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
import json
import shutil

class TrainerRender:

    def __init__(self, config):
        self.config = config
        self.config.near_plane = self.config.near_plane / self.config.scene_scale
        self.config.far_plane = self.config.far_plane / self.config.scene_scale
        
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

        # Load pretrained mesh auto-encoder
        assert self.config.pretrained_mesh_ae is not None
        self.decoder_fn = PointDecoder(config.mesh_ae_hidden_size, voxel_size=1.).to(self.device)
        decoder_dict = torch.load(self.config.pretrained_mesh_ae)
        self.decoder_fn.load_state_dict(decoder_dict['decoder_state_dict'])
        self.decoder_fn.requires_grad = False
        
        # Load pretrained geometry
        assert len(self.config.pretrained_geo_list) > 0 # 1: single scale 2: multi scale (only support 2 scales in the cuurent version)
        assert len(self.config.pretrained_geo_list) == len(self.config.voxel_size_list)
        mesh_embeddings_list = []
        voxel_centers_list = []
        for pretrained_geo in self.config.pretrained_geo_list:

            geo_dict = torch.load(pretrained_geo)['state_dict']

            mesh_embeddings = geo_dict['mesh.mesh_embeddings'].to(self.device).data
            mesh_embeddings = mesh_embeddings / (torch.norm(mesh_embeddings, dim=1, keepdim=True) + 1e-8)
            norms = torch.norm(mesh_embeddings, dim=1)

            voxel_centers = geo_dict['mesh.voxel_centers'].data
            voxel_centers_offset = geo_dict['mesh.voxel_centers_offset'].data
            voxel_centers = (voxel_centers + voxel_centers_offset).to(self.device)
            voxel_centers = voxel_centers[~torch.isnan(norms)]
            voxel_centers = voxel_centers / self.config.scene_scale
            mesh_embeddings = mesh_embeddings[~torch.isnan(norms)]
            mesh_embeddings_list.append(mesh_embeddings)
            voxel_centers_list.append(voxel_centers)

        mesh_embeddings = mesh_embeddings_list
        voxel_centers = voxel_centers_list
        voxel_size = [float(v) / self.config.scene_scale for v in self.config.voxel_size_list]
        
        self.render = DNMPRender(
            self.config, 
            self.device, 
            voxel_size, 
            voxel_centers, 
            mesh_embeddings, 
            self.decoder_fn).to(self.device)
        
        config_dict = vars(self.config)
        config_file = os.path.join(self.config.log_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=1)

        snapshot_dir = os.path.join(self.config.log_dir, 'snapshot')
        ensure_dir(snapshot_dir)

        BASE_DIR = os.getcwd()
        
        # copy core files for repeatability
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
        psnr_meter = AverageMeter()

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
                intrinsic = sample['intrinsic'].squeeze().numpy()
                img_hw = sample['img_hw'].squeeze().numpy()

                if 'depth' in sample.keys() and self.config.use_depth:
                    depth = sample['depth'].squeeze() / self.config.scene_scale
                    depth = depth.to(self.device)

                pose_ = np.eye(4)
                pose_[:3, :4] = pose
                # recenter and rescale poses
                pose = np.matmul(self.transform_M, pose_)
                pose[:3,3] = pose[:3,3] / self.config.scene_scale

                rays_o, rays_d = get_rays(intrinsic, pose, img_hw)

                rays_o = torch.from_numpy(rays_o).float().to(self.device)
                rays_d = torch.from_numpy(rays_d).float().to(self.device)
                if not torch.is_tensor(rgb):
                    rgb = torch.from_numpy(rgb).float()
                rgb = rgb.to(self.device)

                camera = get_camera(intrinsic, pose, img_hw, self.device)

                rays_o_, rays_d_, rgb_, pix_coords_idx = sample_rays(rays_o, rays_d, rgb, self.config.num_rays)
                pix_coords_idx = torch.from_numpy(pix_coords_idx).long().to(self.device)

                if self.config.render_multi_scale:
                    ret_dict = self.render.render_rays_multiscale(rays_o_, rays_d_, pix_coords_idx, camera, img_hw=img_hw)
                else:
                    ret_dict = self.render.render_rays(rays_o_, rays_d_, pix_coords_idx, camera, img_hw=img_hw)
                
                valid_idx = ret_dict['valid_idx']
                loss = ((ret_dict['rgb'][...,:3] - rgb_) ** 2)[valid_idx].mean()
                mse = loss.detach().cpu().numpy()

                # if use_depth is True, use depth for additional supervision
                if 'depth' in sample.keys() and self.config.use_depth:
                    depth_ = depth.view(-1)[pix_coords_idx]
                    depth_loss = torch.abs(ret_dict['depth'] - depth_)
                    loss = loss + depth_loss * 0.01 * self.config.scene_scale
                
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                psnr = metrics.mse2psnr_np(mse)
                loss_meter.update(loss.item())
                psnr_meter.update(psnr)

                if self.global_step % self.config.print_freq == 0:
                    self.logger.info(
                        'Iter [{}/{}] Loss: {:.4f} PSNR: {:.4f}'.format(
                            self.global_step, self.max_iter, loss_meter.avg, psnr_meter.avg))
                    self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                    self.writer.add_scalar('train/psnr', psnr_meter.avg, self.global_step)
                    loss_meter.reset()
                    psnr_meter.reset()
                
                if self.global_step % self.config.val_freq == 0:
                    self.render.eval()
                    self.validate()
                    self.render.train()
                    self.render.config.perturb = self.config.perturb
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
            pose[:3,3] = pose[:3,3] / self.config.scene_scale
            rays_o, rays_d = get_rays(intrinsic, pose, img_hw)

            rays_o = torch.from_numpy(rays_o).float().to(self.device)
            rays_d = torch.from_numpy(rays_d).float().to(self.device)

            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)

            camera = get_camera(intrinsic, pose, img_hw, self.device)
            pix_coords_idx = torch.arange(0, rays_o.shape[0]).long().to(self.device)
            
            if self.config.render_multi_scale:
                ret_dict = self.render.inference_img_multi_scale(rays_o, rays_d, camera, img_hw=img_hw)
            else:
                ret_dict = self.render.inference_img(rays_o, rays_d, camera, img_hw=img_hw)

            pred_rgb = ret_dict['rgb'][...,:3].view(int(img_hw[0]), int(img_hw[1]), 3)
            pred_rgb = pred_rgb.cpu().numpy()
            rgb = rgb.cpu().numpy()

            pred_depth = ret_dict['depth'].view(int(img_hw[0]), int(img_hw[1]))
            pred_depth = pred_depth.cpu().numpy()

            valid_idx = ret_dict['valid_idx'].cpu().numpy()

            mse = ((pred_rgb - rgb) ** 2).reshape(-1, 3)[valid_idx].mean()
            psnr = metrics.mse2psnr_np(mse)
            ssim = metrics.ssim_fn(pred_rgb, rgb)
    
        self.logger.info(f'PSNR: {psnr} SSIM: {ssim}')

        self.writer.add_scalar('val/psnr', psnr, self.val_step)
        self.writer.add_scalar('val/ssim', ssim, self.val_step)

        self.writer.add_image('val/pred_image', pred_rgb, self.val_step, dataformats='HWC')
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