import torch
import numpy as np
import os
import cv2
import time
import sys

from core.cameras import get_camera
from core.train_utils import ensure_dir
import core.metrics as metrics
from datasets import dataset_dict
from tqdm import tqdm

from models.renderer import DNMPRender
from models.mipnerf.mipnerf_render import MipNerfRender
from models.mesh_AE import PointDecoder
from models.ray_utils import get_rays_radii

class Tester:
    def __init__(self, config):
        self.config = config

        self.config.near_plane = self.config.near_plane / self.config.scene_scale
        self.config.far_plane = self.config.far_plane / self.config.scene_scale

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            raise ValueError('GPU not available')
        
        self.save_dir = self.config.save_dir
        ensure_dir(self.save_dir)

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
        
        assert (self.config.pretrained_render is not None)
        render_state_dict = torch.load(self.config.pretrained_render)['state_dict']
        self.render.load_state_dict(render_state_dict)

        self.mipnerf = MipNerfRender(self.config).to(self.device)
        mipnerf_state_dict = torch.load(self.config.pretrained_mipnerf)['state_dict']
        self.mipnerf.load_state_dict(mipnerf_state_dict)

        self.get_dataloader()
    
    def get_dataloader(self):
        dataset = dataset_dict[self.config.dataset]

        self.test_dataset = dataset(self.config, phase='val')

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.config.num_workers,
                                                 pin_memory=False)
    
    def test(self):
        self.render.eval()
        self.mipnerf.eval()

        img_save_dir = os.path.join(self.save_dir, 'pred_imgs')
        ensure_dir(img_save_dir)

        psnrs = []
        ssims = []
        lpipses = []

        metric_fn = os.path.join(self.save_dir, 'metrics.txt')
        metric_f = open(metric_fn, 'w')

        with torch.no_grad():

            for idx, sample in tqdm(enumerate(self.test_loader)):

                rgb = sample['rgb'].squeeze()
                pose = sample['pose'].squeeze().numpy()
                intrinsic = sample['intrinsic'].squeeze().numpy()
                img_hw = sample['img_hw'].squeeze().numpy()
                img_name = sample['img_name'][0]

                pose_ = np.eye(4)
                pose_[:3, :4] = pose
                pose = np.matmul(self.transform_M, pose_)
                pose[:3,3] = pose[:3,3] / self.config.scene_scale
                rays_o, rays_d, radii = get_rays_radii(intrinsic, pose, img_hw)

                rays_o = torch.from_numpy(rays_o).float().to(self.device)
                rays_d = torch.from_numpy(rays_d).float().to(self.device)
                radii = torch.from_numpy(radii).float().to(self.device)

                rays_o = rays_o.view(-1, 3)
                rays_d = rays_d.view(-1, 3)
                radii = radii.view(-1, 1)

                camera = get_camera(intrinsic, pose, img_hw, self.device)
                pix_coords_idx = torch.arange(0, rays_o.shape[0]).long().to(self.device)

                if self.config.render_multi_scale:
                    ret_dict = self.render.inference_img_multi_scale(rays_o, rays_d, camera, img_hw=img_hw)
                else:
                    ret_dict = self.render.inference_img(rays_o, rays_d, camera, img_hw=img_hw)
                
                pred_rgb = ret_dict['rgb'][...,:3]
                pred_rgb = pred_rgb.cpu().numpy()
                rgb = rgb.cpu().numpy().reshape(int(img_hw[0]), int(img_hw[1]), 3)

                pred_depth = ret_dict['depth']
                pred_depth = pred_depth.cpu().numpy()

                valid_idx = ret_dict['valid_idx'].cpu().numpy()

                pix_coords_idx = pix_coords_idx[~valid_idx]
                rays_o = rays_o[~valid_idx]
                rays_d = rays_d[~valid_idx]
                radii = radii[~valid_idx]

                ret_dict_nerf = self.mipnerf.render_chunk(rays_o, rays_d, radii, chunk_size=self.config.chunk_size)
                
                pred_rgb[~valid_idx] = ret_dict_nerf['rgb_fine'][...,:3].cpu().numpy()
                pred_depth[~valid_idx] = ret_dict_nerf['depth_fine'].cpu().numpy()

                pred_rgb = pred_rgb.reshape(int(img_hw[0]), int(img_hw[1]), 3)
                mse = np.mean((pred_rgb - rgb) ** 2)
                psnr = metrics.mse2psnr_np(mse)
                ssim = metrics.ssim_fn(pred_rgb, rgb)
                lpips = metrics.lpips_fn(pred_rgb, rgb)
                psnrs.append(psnr)
                ssims.append(ssim)
                lpipses.append(lpips)

                print(f'{idx} {psnr} {ssim} {lpips}')
                sys.stdout.flush()
                
                metric_f.write(f'{idx} {psnr} {ssim} {lpips} \n')

                save_img_name = os.path.join(img_save_dir, f'{img_name}.png')
                pred_rgb_bgr = cv2.cvtColor(pred_rgb * 255., cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_name, pred_rgb_bgr)
            
            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpipses = np.array(lpipses)
            avg_psnr = np.mean(psnrs)
            avg_ssim = np.mean(ssims)
            avg_lpips = np.mean(lpipses)

            print('Avg psnr:', avg_psnr)
            print('Avg ssim:', avg_ssim)
            print('Avg lpips:', avg_lpips)

            metric_f.write(f'Avg psnr: {avg_psnr}\n')
            metric_f.write(f'Avg ssim: {avg_ssim}\n')
            metric_f.write(f'Avg lpips: {avg_lpips}\n')
            metric_f.close()