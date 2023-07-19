import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsparse.utils.quantize import sparse_quantize
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from fairnr.clib import aabb_ray_intersect

from models.mesh import DNMPs, DNMPScene
from models.utils import PositionEncoding, MLPs, BackgroundMLP
from core.losses import deform_regularization
import numpy as np

class DNMPGeometry(nn.Module):
    def __init__(self,
                 config,
                 device,
                 pts):
        super(DNMPGeometry, self).__init__()

        self.config = config
        self.device = device
        self.voxel_size = config.voxel_size
        
        self.voxel_centers, self.voxel_indices = self.get_voxel_from_pts(self.voxel_size, pts)
        
        self.mesh = DNMPs(
            config=config,
            voxel_centers=self.voxel_centers,
            voxel_size=self.voxel_size,
            device=self.device,
            num_faces=1,
            mesh_embedding_dim=config.mesh_ae_hidden_size)
    
    def get_voxel_from_pts(self, voxel_size, pts):
        """
        Input:
            voxel_size
            pts: np.array (N, 3)
        Output:
            voxel_centers
            voxel_indices
        """
        coords, indices = sparse_quantize(pts,
                                          voxel_size,
                                          return_index=True)
        unique_pts = pts[indices]
        voxel_indices = np.floor(unique_pts / voxel_size)
        if self.config.use_voxel_center:
            voxel_centers = voxel_indices * voxel_size
        else:
            voxel_centers = unique_pts
        
        voxel_indices = torch.from_numpy(voxel_indices).long().to(self.device)
        voxel_centers = torch.from_numpy(voxel_centers).float().to(self.device)

        return voxel_centers, voxel_indices
    
    def ray_intersect(self, ray_start, ray_dir):
        '''
        Inputs:
            origin: [3]
            ray_start: [num_rays, 3] bs can be seen of num of rendered images, only support 1 in the current version
            ray_dir: [num_rays, 3]
        Ret:
            hits: [num_rays, 1]
            min_depth: [num_rays, max_hits, 1]
            max_depth: [num_rays, max_hits, 1]
        '''

        if ray_start.shape[0] != 1:
            ray_start = ray_start.unsqueeze(0)
        if ray_dir.shape[0] != 1:
            ray_dir = ray_dir.unsqueeze(0)

        voxel_centers = self.voxel_centers

        pts_idx, min_depth, max_depth = aabb_ray_intersect(
            self.voxel_size, self.config.max_hits, voxel_centers.unsqueeze(0), ray_start, ray_dir)
        
        MAX_DEPTH = 10000.0
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH) # [1, num_rays, max_hits]
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH) # [1, num_rays, max_hits]
        min_depth, sorted_idx = min_depth.sort(dim=-1) # [1, num_rays, max_hits]
        max_depth = max_depth.gather(-1, sorted_idx) # [1, num_rays, max_hits]
        pts_idx = pts_idx.gather(-1, sorted_idx) # [1, num_rays, max_hits]
        hits = pts_idx.ne(-1).any(-1) # [1, num_rays]

        pts_idx = pts_idx.squeeze(0)
        min_depth = min_depth.squeeze(0)
        max_depth = max_depth.squeeze(0)
        hits = hits.squeeze(0)

        pts_nearest_idx = pts_idx[..., 0:1]
        pts_nearest_idx = pts_nearest_idx.expand_as(pts_idx)
        pts_idx = torch.where(pts_idx==-1, pts_nearest_idx, pts_idx)

        min_depth_first = min_depth[..., 0:1]
        depth_bias = min_depth - min_depth_first
        min_depth_first = min_depth_first.expand_as(min_depth)
        max_depth_first = max_depth[..., 0:1]
        max_depth_first = max_depth_first.expand_as(max_depth)
        min_depth = torch.where(depth_bias < self.config.voxel_depth_range_thresh, min_depth, min_depth_first)
        max_depth = torch.where(depth_bias < self.config.voxel_depth_range_thresh, max_depth, max_depth_first)
        pts_idx = torch.where(depth_bias < self.config.voxel_depth_range_thresh, pts_idx, pts_nearest_idx)

        ray_min_depth = torch.min(min_depth, dim=-1, keepdim=True)[0]
        max_depth_ = max_depth.masked_fill(pts_idx.eq(-1), -MAX_DEPTH)
        ray_max_depth = torch.max(max_depth_, dim=-1, keepdim=True)[0]

        ret_dict = {}
        ret_dict['hits'] = hits # [num_rays]
        ret_dict['ray_min_depth'] = ray_min_depth # [num_rays, 1]
        ret_dict['ray_max_depth'] = ray_max_depth # [num_rays, 1]
        ret_dict['min_depth'] = min_depth # [num_rays, 1]
        ret_dict['max_depth'] = max_depth # [num_rays, 1]
        ret_dict['pts_idx'] = pts_idx # [num_rays, 1]

        return ret_dict
    
    def render_depth_normal(self, rays_o, rays_d, camera, pix_coords, decoder_fn=None, img_hw=None):
        
        intersect_dict = self.ray_intersect(rays_o, rays_d)
        all_idx = torch.arange(0, rays_o.shape[0]).to(rays_o.device).long()
        ray_intersected_idx = all_idx[intersect_dict['hits']]

        voxel_idx_intersected = intersect_dict['pts_idx'][ray_intersected_idx].view(-1)
        unique_voxel_idx, inverse_indices = torch.unique(voxel_idx_intersected, return_inverse=True)
        pix_coords_intersected = pix_coords[ray_intersected_idx]

        mesh_verts, mesh_depth, mesh_normals, meshes = self.mesh.render_depth_and_normal(camera, unique_voxel_idx, pix_coords_intersected, decoder_fn=decoder_fn, img_hw=img_hw)

        if hasattr(self.mesh, 'voxel_centers_offset'):
            offset_thresh = self.voxel_size
            deform_loss = deform_regularization(self.mesh.voxel_centers_offset, thresh=offset_thresh)
        else:
            deform_loss = 0.
        
        mesh_laplacian_loss = mesh_laplacian_smoothing(meshes)
        mesh_normal_loss = mesh_normal_consistency(meshes)

        depth = torch.zeros_like(rays_o[...,0]).to(rays_o)
        depth[ray_intersected_idx] = mesh_depth
        normal = torch.ones_like(rays_o).to(rays_o)
        normal[ray_intersected_idx] = mesh_normals

        ret_dict = {}
        ret_dict['depth'] = depth
        ret_dict['normal'] = normal
        ret_dict['mesh_loss'] = mesh_laplacian_loss + mesh_normal_loss
        ret_dict['deform_loss'] = deform_loss

        return ret_dict


class DNMPRender(nn.Module):

    def __init__(self,
                 config,
                 device,
                 voxel_size,
                 voxel_centers,
                 mesh_embeddings,
                 decoder_fn=None):
        super(DNMPRender, self).__init__()

        self.config = config
        self.device = device
        self.use_xyz_pos = config.use_xyz_pos
        self.use_viewdirs = config.use_viewdirs

        self.render_multi_scale = config.render_multi_scale

        if self.render_multi_scale:
            assert len(voxel_centers) > 1
            assert len(voxel_centers) == len(mesh_embeddings)
            assert len(voxel_centers) == len(voxel_size)

            self.coarse_mesh = DNMPScene(
                config,
                voxel_centers=voxel_centers[1],
                voxel_size=voxel_size[1],
                mesh_embeddings=mesh_embeddings[1],
                device=device,
                num_faces=self.config.coarse_num_faces,
                decoder_fn=decoder_fn,
                vertex_embedding_dim=config.vertex_feats_dim)
            self.fine_mesh = DNMPScene(
                config,
                voxel_centers=voxel_centers[0],
                voxel_size=voxel_size[0],
                mesh_embeddings=mesh_embeddings[0],
                device=device,
                num_faces=self.config.num_faces,
                decoder_fn=decoder_fn,
                vertex_embedding_dim=config.vertex_feats_dim)
            vertex_feats_dim = self.coarse_mesh.vertex_embeddings.shape[-1]
        else:
            self.mesh = DNMPScene(
                config,
                voxel_centers=voxel_centers,
                voxel_size=voxel_size,
                mesh_embeddings=mesh_embeddings,
                device=device,
                num_faces=self.config.num_faces,
                decoder_fn=decoder_fn,
                vertex_embedding_dim=config.vertex_feats_dim)
            vertex_feats_dim = self.mesh.vertex_embeddings.shape[-1]

        self.verts_pos_encoder = PositionEncoding(vertex_feats_dim, config.N_freqs_feats, logscale=config.logscale)

        in_channels = vertex_feats_dim*(2*6+1)
        rgb_in_channels = config.mesh_net_width
        if self.use_xyz_pos:
            self.xyz_pos_encoder = PositionEncoding(3, config.N_freqs_xyz, logscale=config.logscale)
            in_channels += 3*(2*config.N_freqs_xyz+1)
        if self.use_viewdirs:
            self.dir_pos_encoder = PositionEncoding(3, config.N_freqs_dir, logscale=config.logscale)
            rgb_in_channels += 3*(2*config.N_freqs_dir+1)
        
        self.base_embedding_layer = MLPs(
            net_depth=config.mesh_net_depth,
            net_width=config.mesh_net_width,
            in_channels=in_channels,
            skips=[4])
        
        self.opacity_layer = nn.Linear(config.mesh_net_width, 1)
        
        self.rgb_dir_embedding_layer = MLPs(
            net_depth=2,
            net_width=config.mesh_net_width,
            in_channels=rgb_in_channels,
            skips=[])
        
        self.rgb_layer = nn.Linear(config.mesh_net_width, 3)
        
        if self.config.use_bkgd:
            self.bkgd_layer = BackgroundMLP(
                net_depth=2,
                net_width=config.mesh_net_width,
                in_channels_dir=3*(2*config.N_freqs_dir+1))
    
    def render_mesh_opacity_and_rgbs(self, sampled_embeddings, sampled_pts, directions, surface_normals):
        '''
        Input:
            sampled_embeddings: [num_rays, num_faces, C]
            sampled_pts: [num_rays, num_faces, 3]
            directions: [num_rays, 3]
            surface_normals: [num_rays, num_faces, 3]
        Output:
            opacity: [num_rays, num_faces]
            rgb: [num_rays, num_faces, 3]
        '''
        num_rays, num_faces, _ = sampled_embeddings.shape
        feats_embedding = self.verts_pos_encoder(sampled_embeddings)

        if self.use_xyz_pos:
            pts_embedding = self.xyz_pos_encoder(sampled_pts)
            embeddings = torch.cat([feats_embedding, pts_embedding], -1)
        else:
            embeddings = feats_embedding

        embeddings = self.base_embedding_layer(embeddings)

        opacities = self.opacity_layer(embeddings)
        opacities = torch.sigmoid(opacities)

        if self.use_viewdirs and (surface_normals is not None):
            viewdirs = directions / (torch.norm(directions, dim=-1, keepdim=True)+1e-10)
            surface_normals = surface_normals.view(num_rays, num_faces, 3)
            viewdirs = viewdirs.unsqueeze(1).repeat(1,num_faces,1)
            view_normal = surface_normals * viewdirs
            view_normal_embedding = self.dir_pos_encoder(view_normal)
            embeddings = torch.cat([embeddings, view_normal_embedding], -1)
        
        embeddings = self.rgb_dir_embedding_layer(embeddings)
        rgbs = self.rgb_layer(embeddings)
        rgbs = torch.sigmoid(rgbs)

        return opacities, rgbs
    
    def opacity_volumetric_rendering(self, opacities, rgbs, z_vals):
        '''
        Alpha compositing
        Input:
            opacities: [num_rays, num_samples, 1]
            rgbs: [num_rays, num_samples, 3]
            z_vals: [num_rays, num_samples]
        Output:
            rgb_final: [num_rays, 3]
            depth_final: [num_rays]
            weights: [num_rays, num_samples]
            weights_sum: [num_rays], accumulated opacity
        '''
        alphas = opacities.squeeze(-1) # [N_rays, N_samples]
        # shift alphas to [1, 1-a1, 1-a2, ...]
        alpha_shifted = torch.cat([torch.ones_like(alphas[:,:1]),1-alphas+1e-10], -1) # [N_rays, N_samples]

        # T = exp(-sum(delta*sigma)) = cumprod(exp(-delta*sigma)) = cumprod(1-alpha)
        # weight = T * (1-exp(-delta*sigma)) = T * alphas
        weights = alphas * torch.cumprod(alpha_shifted, -1)[:,:-1] # [N_rays, N_samples]
        weights_sum = weights.sum(1) # [N_rays]

        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # [N_rays, 3]
        depth_final = torch.sum(weights*z_vals, -1) # [N_rays]

        return rgb_final, depth_final, weights, weights_sum
    
    def render_rays(self, rays_o, rays_d, pix_coords, cameras, img_hw):
        '''
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            pix_coords: [N_rays,]
            cameras: camera object of pytorch3d
        '''

        all_idx = torch.arange(0, rays_o.shape[0]).to(rays_o.device).long()

        sampled_embeddings, sampled_pts, sampled_depth, sampled_normal = self.mesh.render_rays(cameras, pix_coords, img_hw=img_hw)
        valid_idx = (sampled_depth[...,0] > 0)
        sampled_embeddings = sampled_embeddings[valid_idx]
        sampled_pts = sampled_pts[valid_idx]
        sampled_depth = sampled_depth[valid_idx]
        sampled_normal = sampled_normal[valid_idx]

        ray_intersected_idx = all_idx[valid_idx]
        ray_miss_idx = all_idx[~valid_idx]

        rays_o_miss = rays_o[ray_miss_idx]
        rays_d_miss = rays_d[ray_miss_idx]
        rays_o_intersected = rays_o[ray_intersected_idx]
        rays_d_intersected = rays_d[ray_intersected_idx]

        rgb_out = torch.ones(rays_o.shape[0], 3).float().to(rays_o)
        depth_out = torch.ones_like(rays_o[...,0]).to(rays_o)
        rgb_out_coarse = torch.ones(rays_o.shape[0], 3).float().to(rays_o)
        depth_out_coarse = torch.ones_like(rays_o[...,0]).to(rays_o)

        # if hit meshes
        if rays_o_miss.shape[0] != rays_o.shape[0]:
            opacities_intersected, rgbs_intersected = self.render_mesh_opacity_and_rgbs(sampled_embeddings, sampled_pts, rays_d_intersected, sampled_normal)
            rgb_intersected, depth_intersected, weights_intersected, alpha_intersected = self.opacity_volumetric_rendering(opacities_intersected, rgbs_intersected, sampled_depth, rays_d_intersected)
            if self.config.use_bkgd:
                view_dirs_embedding = self.dir_pos_encoder(rays_d_intersected / (torch.norm(rays_d_intersected, dim=-1, keepdim=True)+1e-8))
                bkgd_rgb = self.background_nerf(view_dirs_embedding)
                rgb_intersected = rgb_intersected + (1 - bkgd_rgb.unsqueeze(-1)) * bkgd_rgb
            rgb_out[ray_intersected_idx] = rgb_intersected
            depth_out[ray_intersected_idx] = depth_intersected
        
        ret_dict = {}
        ret_dict['rgb'] = rgb_out
        ret_dict['depth'] = depth_out
        ret_dict['rgb_coarse'] = rgb_out_coarse
        ret_dict['depth_coarse'] = depth_out_coarse
        ret_dict['ray_intersected_idx'] = ray_intersected_idx.squeeze()
        ret_dict['ray_miss_idx'] = ray_miss_idx.squeeze()
        ret_dict['valid_idx'] = valid_idx

        return ret_dict
    
    def render_rays_multiscale(self, rays_o, rays_d, pix_coords, cameras, img_hw=None):
        '''
            Render rays with multiscale mesh, only support 2 scales in the current version
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            pix_coords: [N_rays,]
            cameras: camera object of pytorch3d
        '''
        all_idx = torch.arange(0, rays_o.shape[0]).to(rays_o.device).long()

        coarse_embeddings, coarse_pts, coarse_depth, coarse_normal = self.coarse_mesh.render_rays(cameras, pix_coords, img_hw=img_hw)
        fine_embeddings, fine_pts, fine_depth, fine_normal = self.fine_mesh.render_rays(cameras, pix_coords, img_hw=img_hw)
        
        coarse_valid_idx = (coarse_depth[...,0] > 0)
        fine_valid_idx = (fine_depth[...,0] > 0)
        valid_idx = (coarse_valid_idx) | (fine_valid_idx)

        coarse_embeddings = coarse_embeddings[valid_idx]
        coarse_pts = coarse_pts[valid_idx]
        coarse_depth = coarse_depth[valid_idx]
        coarse_normal = coarse_normal[valid_idx]

        fine_embeddings = fine_embeddings[valid_idx]
        fine_pts = fine_pts[valid_idx]
        fine_depth = fine_depth[valid_idx]
        fine_normal = fine_normal[valid_idx]
        
        ray_intersected_idx = all_idx[valid_idx]
        ray_miss_idx = all_idx[~valid_idx]

        rays_o_miss = rays_o[ray_miss_idx]
        rays_d_miss = rays_d[ray_miss_idx]
        rays_o_intersected = rays_o[ray_intersected_idx]
        rays_d_intersected = rays_d[ray_intersected_idx]

        rgb_out = torch.ones(rays_o.shape[0], 3).float().to(rays_o)
        depth_out = torch.zeros_like(rays_o[...,0]).to(rays_o)
        rgb_out_coarse = torch.ones(rays_o.shape[0], 3).float().to(rays_o)
        depth_out_coarse = torch.zeros_like(rays_o[...,0]).to(rays_o)

        # if hit meshes
        if rays_o_miss.shape[0] != rays_o.shape[0]:
            coarse_opacities_intersected, coarse_rgbs_intersected = self.render_mesh_opacity_and_rgbs(coarse_embeddings, coarse_pts, rays_d_intersected, coarse_normal)
            coarse_rgb_intersected, coarse_depth_intersected, coarse_weights_intersected, coarse_alpha_intersected = \
                self.opacity_volumetric_rendering(coarse_opacities_intersected, coarse_rgbs_intersected, coarse_depth)
            fine_opacities_intersected, fine_rgbs_intersected = self.render_mesh_opacity_and_rgbs(fine_embeddings, fine_pts, rays_d_intersected, fine_normal)
            fine_rgb_intersected, fine_depth_intersected, fine_weights_intersected, fine_alpha_intersected = \
                self.opacity_volumetric_rendering(fine_opacities_intersected, fine_rgbs_intersected, fine_depth)
            fine_alpha_intersected[fine_depth_intersected <= 0] = 0.
            fine_rgb_intersected[fine_depth_intersected <= 0] = 0.
            fine_depth_intersected[fine_depth_intersected <= 0] = 0.
            
            rgb_intersected = (1 - fine_alpha_intersected.unsqueeze(-1)) * coarse_rgb_intersected + fine_rgb_intersected
            depth_intersected = (1 - fine_alpha_intersected) * coarse_depth_intersected + fine_depth_intersected

            if self.config.use_bkgd:
                view_dirs_embedding = self.dir_pos_encoder(rays_d_intersected / (torch.norm(rays_d_intersected, dim=-1, keepdim=True)+1e-8))
                bkgd_rgb = self.bkgd_layer(view_dirs_embedding)
                rgb_intersected = rgb_intersected + (1 - fine_alpha_intersected.unsqueeze(-1)) * (1 - coarse_alpha_intersected.unsqueeze(-1)) * bkgd_rgb
            
            rgb_out[ray_intersected_idx] = rgb_intersected
            depth_out[ray_intersected_idx] = depth_intersected
        
        ret_dict = {}
        ret_dict['rgb'] = rgb_out
        ret_dict['depth'] = depth_out
        ret_dict['rgb_coarse'] = rgb_out_coarse
        ret_dict['depth_coarse'] = depth_out_coarse
        ret_dict['ray_intersected_idx'] = ray_intersected_idx.squeeze()
        ret_dict['ray_miss_idx'] = ray_miss_idx.squeeze()
        ret_dict['valid_idx'] = valid_idx
        return ret_dict
    
    def inference_img(self, rays_o, rays_d, cameras, img_hw=None):
        '''
            rays_o: [H*W,3]
            rays_d: [H*W,3]
        '''
        all_idx = torch.arange(0, rays_o.shape[0]).to(rays_o.device).long()
        pix_coords = all_idx.clone()
        sampled_embeddings, sampled_pts, sampled_depth, sampled_normal = self.mesh.render_rays(cameras, pix_coords, img_hw=img_hw)
        valid_idx = (sampled_depth[...,0] > 0)
        sampled_embeddings = sampled_embeddings[valid_idx]
        sampled_pts = sampled_pts[valid_idx]
        sampled_depth = sampled_depth[valid_idx]
        sampled_normal = sampled_normal[valid_idx]

        ray_intersected_idx = all_idx[valid_idx]
        ray_miss_idx = all_idx[~valid_idx]

        rays_o_miss = rays_o[ray_miss_idx]
        rays_d_miss = rays_d[ray_miss_idx]
        rays_o_intersected = rays_o[ray_intersected_idx]
        rays_d_intersected = rays_d[ray_intersected_idx]

        if self.config.mesh_chunk_size is not None:
            mesh_chunk_size = self.config.mesh_chunk_size
            num_mesh_chunk = int(np.ceil(sampled_embeddings.shape[0] / mesh_chunk_size))
            rgbs_intersected_list = []
            opacities_intersected_list = []
            for i in range(num_mesh_chunk):
                sampled_embeddings_ = sampled_embeddings[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                sampled_pts_ = sampled_pts[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                rays_d_intersected_ = rays_d_intersected[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                sampled_normal_ = sampled_normal[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                opacities_intersected, rgbs_intersected = self.render_mesh_sigmas_and_rgbs(sampled_embeddings_, sampled_pts_, rays_d_intersected_, sampled_normal_)
                rgbs_intersected_list.append(rgbs_intersected)
                opacities_intersected_list.append(opacities_intersected)
            rgbs_intersected = torch.cat(rgbs_intersected_list, dim=0)
            opacities_intersected = torch.cat(opacities_intersected_list, dim=0)
        else:
            opacities_intersected, rgbs_intersected = self.render_mesh_opacity_and_rgbs(sampled_embeddings, sampled_pts, rays_d_intersected, sampled_normal)
        
        rgb_intersected, depth_intersected, weights_intersected, _ = self.opacity_volumetric_rendering(opacities_intersected, rgbs_intersected, sampled_depth)
        
        rgb_out = torch.ones(rays_o.shape[0], 3).float().to(rays_o)
        rgb_out[ray_intersected_idx] = rgb_intersected
        depth_out = torch.zeros_like(rays_o[...,0]).to(rays_o)
        depth_out[ray_intersected_idx] = depth_intersected
        
        ret_dict = {}
        ret_dict['rgb'] = rgb_out
        ret_dict['depth'] = depth_out
        ret_dict['valid_idx'] = valid_idx

        return ret_dict
    
    def inference_img_multi_scale(self, rays_o, rays_d, cameras, img_hw=None):
        '''
            rays_o: [H*W,3]
            rays_d: [H*W,3]
        '''
        all_idx = torch.arange(0, rays_o.shape[0]).to(rays_o.device).long()
        pix_coords = all_idx.clone()

        coarse_embeddings, coarse_pts, coarse_depth, coarse_normal = self.coarse_mesh.render_rays(cameras, pix_coords, img_hw=img_hw)
        fine_embeddings, fine_pts, fine_depth, fine_normal = self.fine_mesh.render_rays(cameras, pix_coords, img_hw=img_hw)
        
        coarse_valid_idx = (coarse_depth[...,0] > 0)
        fine_valid_idx = (fine_depth[...,0] > 0)
        valid_idx = (coarse_valid_idx) | (fine_valid_idx)

        coarse_embeddings = coarse_embeddings[valid_idx]
        coarse_pts = coarse_pts[valid_idx]
        coarse_depth = coarse_depth[valid_idx]
        coarse_normal = coarse_normal[valid_idx]

        fine_embeddings = fine_embeddings[valid_idx]
        fine_pts = fine_pts[valid_idx]
        fine_depth = fine_depth[valid_idx]
        fine_normal = fine_normal[valid_idx]
        
        ray_intersected_idx = all_idx[valid_idx]
        ray_miss_idx = all_idx[~valid_idx]

        rays_o_miss = rays_o[ray_miss_idx]
        rays_d_miss = rays_d[ray_miss_idx]
        rays_o_intersected = rays_o[ray_intersected_idx]
        rays_d_intersected = rays_d[ray_intersected_idx]

        if self.config.mesh_chunk_size is not None:
            mesh_chunk_size = self.config.mesh_chunk_size
            num_mesh_chunk = int(np.ceil(coarse_embeddings.shape[0] / mesh_chunk_size))
            coarse_rgbs_intersected_list = []
            coarse_opacities_intersected_list = []
            fine_rgbs_intersected_list = []
            fine_opacities_intersected_list = []
            for i in range(num_mesh_chunk):
                coarse_embeddings_ = coarse_embeddings[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                coarse_pts_ = coarse_pts[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                coarse_sampled_normal_ = coarse_normal[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                fine_embeddings_ = fine_embeddings[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                fine_pts_ = fine_pts[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                fine_sampled_normal_ = fine_normal[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                rays_d_intersected_ = rays_d_intersected[i*mesh_chunk_size:(i+1)*mesh_chunk_size]
                coarse_opacities_intersected, coarse_rgbs_intersected = self.render_mesh_opacity_and_rgbs(coarse_embeddings_, coarse_pts_, rays_d_intersected_, coarse_sampled_normal_)
                fine_opacities_intersected, fine_rgbs_intersected = self.render_mesh_opacity_and_rgbs(fine_embeddings_, fine_pts_, rays_d_intersected_, fine_sampled_normal_)
                coarse_opacities_intersected_list.append(coarse_opacities_intersected)
                coarse_rgbs_intersected_list.append(coarse_rgbs_intersected)
                fine_opacities_intersected_list.append(fine_opacities_intersected)
                fine_rgbs_intersected_list.append(fine_rgbs_intersected)
            coarse_opacities_intersected = torch.cat(coarse_opacities_intersected_list, 0)
            coarse_rgbs_intersected = torch.cat(coarse_rgbs_intersected_list, 0)
            fine_opacities_intersected = torch.cat(fine_opacities_intersected_list, 0)
            fine_rgbs_intersected = torch.cat(fine_rgbs_intersected_list, 0)
        else:
            coarse_opacities_intersected, coarse_rgbs_intersected = self.render_mesh_opacity_and_rgbs(coarse_embeddings, coarse_pts, rays_d_intersected, coarse_normal)
            fine_opacities_intersected, fine_rgbs_intersected = self.render_mesh_opacity_and_rgbs(fine_embeddings, fine_pts, rays_d_intersected, fine_normal)

        coarse_rgb_intersected, coarse_depth_intersected, coarse_weights_intersected, coarse_alpha_intersected = \
            self.opacity_volumetric_rendering(coarse_opacities_intersected, coarse_rgbs_intersected, coarse_depth)
        fine_rgb_intersected, fine_depth_intersected, fine_weights_intersected, fine_alpha_intersected = \
            self.opacity_volumetric_rendering(fine_opacities_intersected, fine_rgbs_intersected, fine_depth)

        fine_alpha_intersected[fine_depth_intersected <= 0] = 0.
        fine_rgb_intersected[fine_depth_intersected <= 0] = 0.
        fine_depth_intersected[fine_depth_intersected <= 0] = 0.
        
        rgb_intersected = (1 - fine_alpha_intersected.unsqueeze(-1)) * coarse_rgb_intersected + fine_rgb_intersected
        depth_intersected = (1 - fine_alpha_intersected) * coarse_depth_intersected + fine_depth_intersected

        if self.config.use_bkgd:
            view_dirs_embedding = self.dir_pos_encoder(rays_d_intersected / (torch.norm(rays_d_intersected, dim=-1, keepdim=True)+1e-8))
            bkgd_rgb = self.bkgd_layer(view_dirs_embedding)
            rgb_intersected = rgb_intersected + (1 - fine_alpha_intersected.unsqueeze(-1)) * (1 - coarse_alpha_intersected.unsqueeze(-1)) * bkgd_rgb
        
        rgb_out = torch.ones(rays_o.shape[0], 3).float().to(rays_o)
        rgb_out[ray_intersected_idx] = rgb_intersected
        depth_out = torch.zeros_like(rays_o[...,0]).to(rays_o)
        depth_out[ray_intersected_idx] = depth_intersected
        
        ret_dict = {}
        ret_dict['rgb'] = rgb_out
        ret_dict['depth'] = depth_out
        ret_dict['valid_idx'] = valid_idx

        return ret_dict