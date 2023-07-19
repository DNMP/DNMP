import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils import PositionEncoding

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous().to(cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

class NerfMLP(nn.Module):
    '''
    Args:
        net_depth: number of layers in the network
        net_width: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz
        in_channels_dir: number of input channels for directions
        skips: list of integers indicating which layers to skip
        use_viewdirs: if True, use view directions as an input to infer RGB
    Inputs:
        x: (batch_size, num_samples, channels) [xyz, directions]
    '''
    def __init__(self, 
                 net_depth=8, 
                 net_width=256,
                 in_channels_xyz=63,
                 in_channels_dir=27,
                 skips=[4],
                 use_viewdirs=True,
                 render_semantic=False,
                 num_semantic=19):
        super(NerfMLP, self).__init__()

        self.net_depth = net_depth
        self.net_width = net_width
        self.use_viewdirs = use_viewdirs
        self.skips = skips

        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.xyz_layers = nn.ModuleList()

        for i in range(self.net_depth):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, self.net_width)
            elif i in skips:
                layer = nn.Linear(self.net_width+in_channels_xyz, self.net_width)
            else:
                layer = nn.Linear(self.net_width, self.net_width)
            self.xyz_layers.append(layer)
        
        self.sigma_layer = nn.Linear(self.net_width, 1)
        self.render_semantic = render_semantic
        if self.render_semantic:
            self.semantic_feature_layer = nn.Linear(self.net_width, self.net_width)
            self.semantic_layer = nn.Linear(self.net_width, num_semantic)

        if self.use_viewdirs:
            self.feature_layer = nn.Linear(self.net_width, self.net_width)
            self.dir_layer = nn.Linear(in_channels_dir+self.net_width, self.net_width//2)
            self.rgb_layer = nn.Linear(self.net_width//2, 3)
        else:
            self.rgb_layer = nn.Linear(self.net_width, 3)
    
    def forward(self, x, use_opacity=False):
        
        input_xyz = x[...,:self.in_channels_xyz]
        input_dir = x[...,self.in_channels_xyz:self.in_channels_xyz+self.in_channels_dir]

        h = input_xyz
        for i, layer in enumerate(self.xyz_layers):
            if i in self.skips:
                h = torch.cat([h, input_xyz], -1)
            h = F.relu(layer(h))
        if use_opacity:
            sigma = torch.sigmoid(self.sigma_layer(h))
        else:
            sigma = F.relu(self.sigma_layer(h))
        
        if self.use_viewdirs:
            feature = self.feature_layer(h)
            h_ = F.relu(self.dir_layer(torch.cat([input_dir, feature], -1)))
            rgb = torch.sigmoid(self.rgb_layer(h_))
        else:
            rgb = torch.sigmoid(self.rgb_layer(h))
        
        if self.render_semantic:
            semantic_feature = self.semantic_feature_layer(h)
            semantic = self.semantic_layer(semantic_feature)
            rgb = torch.cat([rgb, semantic], -1)
        
        rgb_sigma = torch.cat([rgb, sigma], -1)
        
        return rgb_sigma

class NerfRender(nn.Module):
    def __init__(self, config):
        super(NerfRender, self).__init__()

        self.config = config
        self.near_plane = config.near_plane
        self.far_plane = config.far_plane
        self.use_disp = config.use_disp

        self.N_samples_coarse = config.N_samples_coarse
        self.N_samples_fine = config.N_samples_fine

        self.xyz_pos_encoder = PositionEncoding(3, config.N_freqs_xyz, logscale=config.logscale)
        self.dir_pos_encoder = PositionEncoding(3, config.N_freqs_dir, logscale=config.logscale)

        self.coarse_nerf = NerfMLP(net_depth=self.config.nerf_net_depth,
                                   net_width=self.config.nerf_net_width, 
                                   in_channels_xyz=3*(2*config.N_freqs_xyz+1), 
                                   in_channels_dir=3*(2*config.N_freqs_dir+1),
                                   render_semantic=self.config.render_semantic,
                                   num_semantic=self.config.num_semantic)
        self.fine_nerf = NerfMLP(net_depth=self.config.nerf_net_depth, 
                                 net_width=self.config.nerf_net_width,
                                 in_channels_xyz=3*(2*config.N_freqs_xyz+1),
                                 in_channels_dir=3*(2*config.N_freqs_dir+1),
                                 render_semantic=self.config.render_semantic,
                                 num_semantic=self.config.num_semantic)
        
        if self.config.render_semantic:
            self.rgb_out_channel = 3 + self.config.num_semantic
        else:
            self.rgb_out_channel = 3
    
    def sample_rays(self, rays_o, rays_d, near, far, N_samples):
        '''
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            near: [N_rays, 1]
            far: [N_rays, 1]
        '''
        z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device)
        if not self.use_disp:
            z_vals = near * (1 - z_steps) + far * z_steps
        else:
            near = torch.clamp(near, min=0.5)
            z_vals = 1/(1/near* (1 - z_steps) + 1/far * z_steps) # sampling in disparity space
        
        z_vals = z_vals.expand(rays_o.shape[0], N_samples)

        if self.perturb > 0:
            z_vals_mid = 0.5 * (z_vals[:,:-1] + z_vals[:,1:])
            upper = torch.cat([z_vals_mid, z_vals[:,-1:]], -1)
            lower = torch.cat([z_vals[:,:1], z_vals_mid], -1)

            perturb_rand = self.perturb * torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = upper * perturb_rand + lower * (1 - perturb_rand)
        
        xyz_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

        return xyz_sampled, z_vals
    
    def volumetric_rendering(self, sigmas, rgbs, z_vals, rays_d):
        '''
            sigmas: [num_rays, num_samples, 1]
            rgbs: [num_rays, num_samples, 3]
            z_vals: [num_rays, num_samples]
            rays_d: [num_rays, 3]
        '''
        deltas = z_vals[...,1:] - z_vals[...,:-1]
        delta_inf = 1e10 * torch.ones_like(deltas[...,:1]).to(deltas)
        deltas = torch.cat([deltas, delta_inf], -1)

        deltas = deltas * torch.norm(rays_d.unsqueeze(1), dim=-1)
        sigmas = sigmas.squeeze(-1)
        noise = self.perturb * torch.randn(sigmas.shape, device=sigmas.device)

        alphas = 1 - torch.exp(-deltas*torch.relu(sigmas+noise)) # [N_rays, N_samples]
        alphas[..., -1] = 1. # [N_rays, N_samples]
        # shift alphas to [1, 1-a1, 1-a2, ...]
        alpha_shifted = torch.cat([torch.ones_like(alphas[:,:1]),1-alphas+1e-10], -1) # [N_rays, N_samples]

        # T = exp(-sum(delta*sigma)) = cumprod(exp(-delta*sigma)) = cumprod(1-alpha)
        # weight = T * (1-exp(-delta*sigma)) = T * alphas
        weights = alphas * torch.cumprod(alpha_shifted, -1)[:,:-1] # [N_rays, N_samples]
        weights_sum = weights.sum(1) # [N_rays]

        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # [N_rays, 3]
        depth_final = torch.sum(weights*z_vals, -1) # [N_rays]

        return rgb_final, depth_final, weights
    
    def basic_render(self, xyz_samples, rays_d, render_fn, use_opacity=False):
        '''
            xyz_samples: [num_rays, num_samples, 3]
            rays_d: [num_rays, 3]
        '''
        xyz_embedding = self.xyz_pos_encoder(xyz_samples)
        view_dirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True)+1e-5)
        dir_embedding = self.dir_pos_encoder(view_dirs)
        dir_embedding = dir_embedding.unsqueeze(1).repeat(1, xyz_samples.shape[1], 1)

        inputs = (torch.cat([xyz_embedding, dir_embedding], -1)).float()
        rgb_sigma = render_fn(inputs, use_opacity)
        rgb = rgb_sigma[...,:self.rgb_out_channel]
        sigma = rgb_sigma[...,self.rgb_out_channel:]

        return sigma, rgb
    
    def render_rays(self, rays_o, rays_d):
        min_depth = self.config.near_plane * torch.ones_like(rays_o[...,:1]).to(rays_o)
        max_depth = self.config.far_plane * torch.ones_like(rays_o[...,:1]).to(rays_o)

        xyz_samples, z_vals = self.sample_rays(rays_o, rays_d, min_depth, max_depth, self.N_samples_coarse)

        coarse_sigma, coarse_rgb = self.basic_render(xyz_samples, rays_d, render_fn=self.coarse_nerf)

        rgb_coarse, depth_coarse, weights_coarse = self.volumetric_rendering(coarse_sigma, coarse_rgb, z_vals, rays_d)

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights_coarse[...,1:-1], self.N_samples_fine, det=(self.config.perturb==0.), pytest=False)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        xyz_samples = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        fine_sigma, fine_rgb = self.basic_render(xyz_samples, rays_d, render_fn=self.fine_nerf)
        rgb_fine, depth_fine, weights_fine = self.volumetric_rendering(fine_sigma, fine_rgb, z_vals, rays_d)

        ret_dict_nerf = {}
        ret_dict_nerf['rgb_coarse'] = rgb_coarse
        ret_dict_nerf['rgb_fine'] = rgb_fine
        ret_dict_nerf['depth_coarse'] = depth_coarse
        ret_dict_nerf['depth_fine'] = depth_fine

        return ret_dict_nerf