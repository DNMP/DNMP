import torch
import numpy as np
import os
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from meshae.datasets import MeshDataset
import torch.nn.functional as F
from core.train_utils import ensure_dir, get_logger, AverageMeter
from autoencoder import PointEncoder, PointDecoder
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.utils import ico_sphere

class Trainer:

    def __init__(self, config):
        
        self.config = config
        ensure_dir(self.config.log_dir)
        self.logger = get_logger(os.path.join(self.config.log_dir, 'train.log'))
        self.logger.info(self.config)
        log_name = self.config.log_dir.split('/')[-1]
        self.checkpoint_dir = self.config.checkpoint_dir
        ensure_dir(self.checkpoint_dir)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            raise ValueError('GPU not available')
        
        self.encoder = PointEncoder(out_ch=8).to(self.device)
        self.decoder = PointDecoder(input_ch=8, out_ch=3*42, voxel_size=1.).to(self.device)

        self.optimizer = Adam([{'params': self.encoder.parameters()},
                                {'params': self.decoder.parameters()}], 
                               lr=self.config.lr,
                               betas=(0.9, 0.999),
                               weight_decay=1e-6)
        
        self.scheduler = ExponentialLR(self.optimizer, 0.999999)
        self.global_step = 0
        self.val_step = 0

        self.max_iter = self.config.max_iter
        self.best_val_psnr = 0.0

        self.get_dataloader()
    
    def get_dataloader(self):
        self.train_dataset = MeshDataset(self.config, phase='train', augmentation=True)
        self.val_dataset = MeshDataset(self.config, phase='val', augmentation=False)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.config.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.config.num_workers,
                                                   pin_memory=False, drop_last=True)

        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.config.num_workers,
                                                 pin_memory=False)
    
    def train(self):
        self.logger.info('Training started')
        flag = True
        loss_meter = AverageMeter()

        while flag:

            for idx, sample in enumerate(self.train_loader):

                if self.global_step > self.max_iter:
                    flag = False
                    break

                self.global_step += 1
                self.optimizer.zero_grad()
                verts = sample['vertices'].to(self.device)
                faces = sample['faces'].to(self.device)
                
                normalize = True
                if normalize:
                    verts = verts - torch.mean(verts, dim=1, keepdim=True)
                    verts = verts / torch.max(torch.max(torch.abs(verts), dim=1, keepdim=True)[0], dim=2, keepdim=True)[0]

                # encode
                latent_code = self.encoder(verts)
                # decode
                pred_verts = self.decoder(latent_code)

                # loss
                meshes_gt = Meshes(verts, faces)
                meshes_pred = Meshes(pred_verts, faces)
                num_samples = 500
                sampled_pts_gt = sample_points_from_meshes(meshes_gt, num_samples)
                sampled_pts_pred = sample_points_from_meshes(meshes_pred, num_samples)
                loss_pts, _ = chamfer_distance(sampled_pts_gt, sampled_pts_pred)
                normal_loss_pred = mesh_normal_consistency(meshes_pred)
                smooth_loss_pred = mesh_laplacian_smoothing(meshes_pred)

                template_mesh = ico_sphere(level=1, device=self.device)
                template_verts = template_mesh.verts_packed().to(self.device)
                template_faces = template_mesh.faces_packed().to(self.device).unsqueeze(0)
                scale = 1.
                template_verts = template_verts.unsqueeze(0) * scale
                temp_latent_code = self.encoder(template_verts)
                temp_pred_verts = self.decoder(temp_latent_code)
                meshes_temp_pred = Meshes(temp_pred_verts, template_faces)
                num_samples = 500
                sampled_pts_temp = sample_points_from_meshes(template_mesh, num_samples)
                sampled_pts_temp_pred = sample_points_from_meshes(meshes_temp_pred, num_samples)
                loss_pts_temp, _ = chamfer_distance(sampled_pts_temp, sampled_pts_temp_pred)
                loss = loss_pts + 0.10 * (normal_loss_pred + smooth_loss_pred) + 0.05 * loss_pts_temp

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                loss_meter.update(loss.item())

                if self.global_step % self.config.print_freq == 0:
                    self.logger.info(
                        'Iter [{}/{}] Loss: {:.4f}'.format(
                            self.global_step, self.max_iter, loss_meter.avg))
                    loss_meter.reset()

                if self.global_step % self.config.save_step == 0:
                    self.save_ckpt(self.global_step)
    
    def save_ckpt(self, step):

        self.logger.info(f'Save checkpoint for step {step}.')
        state = {
            'step': step,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'ckpt_{step}.pth')
        torch.save(state, filename)