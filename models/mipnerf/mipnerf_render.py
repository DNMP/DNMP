import torch
import typing
import numpy as np
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

Rays = namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))
Rays_keys = Rays._fields

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))

def lift_gaussian(directions, t_mean, t_var, r_var, diagonal):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = torch.unsqueeze(directions, dim=-2) * torch.unsqueeze(t_mean, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
    d_norm_denominator = torch.sum(directions ** 2, dim=-1, keepdim=True) + 1e-10
    # min_denominator = torch.full_like(d_norm_denominator, 1e-10)
    # d_norm_denominator = torch.maximum(min_denominator, d_norm_denominator)

    if diagonal:
        d_outer_diag = directions ** 2  # eq (16)
        null_outer_diag = 1 - d_outer_diag / d_norm_denominator
        t_cov_diag = torch.unsqueeze(t_var, dim=-1) * torch.unsqueeze(d_outer_diag,
                                                                      dim=-2)  # [B, N, 1] * [B, 1, 3] = [B, N, 3]
        xy_cov_diag = torch.unsqueeze(r_var, dim=-1) * torch.unsqueeze(null_outer_diag, dim=-2)
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = torch.unsqueeze(directions, dim=-1) * torch.unsqueeze(directions,
                                                                        dim=-2)  # [B, 3, 1] * [B, 1, 3] = [B, 3, 3]
        eye = torch.eye(directions.shape[-1], device=directions.device)  # [B, 3, 3]
        # [B, 3, 1] * ([B, 3] / [B, 1])[..., None, :] = [B, 3, 3]
        null_outer = eye - torch.unsqueeze(directions, dim=-1) * (directions / d_norm_denominator).unsqueeze(-2)
        t_cov = t_var.unsqueeze(-1).unsqueeze(-1) * d_outer.unsqueeze(-3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        xy_cov = t_var.unsqueeze(-1).unsqueeze(-1) * null_outer.unsqueeze(
            -3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        cov = t_cov + xy_cov
        return mean, cov

def conical_frustum_to_gaussian(directions, t0, t1, base_radius, diagonal, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `directions` is normalized.
    Args:
        directions: torch.tensor float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diagonal: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                            (3 * mu ** 2 + hw ** 2) ** 2)
        r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                    (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    else:
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    return lift_gaussian(directions, t_mean, t_var, r_var, diagonal)

def cast_rays(t_samples, origins, directions, radii, ray_shape, diagonal=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
        t_samples: float array [B, n_sample+1], the "fencepost" distances along the ray.
        origins: float array [B, 3], the ray origin coordinates.
        directions [B, 3]: float array, the ray direction vectors.
        radii[B, 1]: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diagonal: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_samples[..., :-1]  # [B, n_samples]
    t1 = t_samples[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        raise NotImplementedError
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diagonal)
    means = means + torch.unsqueeze(origins, dim=-2)
    return means, covs

def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, disparity, ray_shape):
    """
    Stratified sampling along the rays.
    Args:
        origins: torch.Tensor, [batch_size, 3], ray origins.
        directions: torch.Tensor, [batch_size, 3], ray directions.
        radii: torch.Tensor, [batch_size, 3], ray radii.
        num_samples: int.
        near: torch.Tensor, [batch_size, 1], near clip.
        far: torch.Tensor, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        disparity: bool, sampling linearly in disparity rather than depth.
        ray_shape: string, which shape ray to assume.
    Returns:
    t_samples: torch.Tensor, [batch_size, num_samples], sampled z values.
    means: torch.Tensor, [batch_size, num_samples, 3], sampled means.
    covs: torch.Tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_samples = torch.linspace(0., 1., num_samples + 1, device=origins.device)

    if disparity:
        t_samples = 1. / (1. / near * (1. - t_samples) + 1. / far * t_samples)
    else:
        # t_samples = near * (1. - t_samples) + far * t_samples
        t_samples = near + (far - near) * t_samples

    if randomized:
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)
        lower = torch.cat([t_samples[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_samples = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_samples to make the returned shape consistent.
        t_samples = torch.broadcast_to(t_samples, [batch_size, num_samples + 1])
    means, covs = cast_rays(t_samples, origins, directions, radii, ray_shape)
    return t_samples, (means, covs)

def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """
    Piecewise-Constant PDF sampling from sorted bins.
    Args:
        bins: torch.Tensor, [batch_size, num_bins + 1].
        weights: torch.Tensor, [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        t_samples: torch.Tensor, [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)  # [B, 1]
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
                     cdf,
                     torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)],
                    dim=-1)  # [B, N]

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(
            to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples

def resample_along_rays(origins, directions, radii, t_samples, weights, randomized, ray_shape, stop_grad,
                        resample_padding):
    """Resampling.
    Args:
        origins: torch.Tensor, [batch_size, 3], ray origins.
        directions: torch.Tensor, [batch_size, 3], ray directions.
        radii: torch.Tensor, [batch_size, 3], ray radii.
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        weights: torch.Tensor [batch_size, num_samples], weights for t_samples
        randomized: bool, use randomized samples.
        ray_shape: string, which kind of shape to assume for the ray.
        stop_grad: bool, whether or not to backprop through sampling.
        resample_padding: float, added to the weights before normalizing.
    Returns:
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        points: torch.Tensor, [batch_size, num_samples, 3].
    """
    # Do a blurpool.
    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_samples,
                weights,
                t_samples.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_samples,
            weights,
            t_samples.shape[-1],
            randomized,
        )
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)


def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * torch.sin(x)  # [B, N, 2*3*L]
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y ** 2
    y_var = torch.maximum(torch.zeros_like(y_var), y_var)
    return y, y_var

def integrated_pos_enc(means_covs, min_deg, max_deg, diagonal=True):
    """Encode `means` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        means_covs:[B, N, 3] a tuple containing: means, torch.Tensor, variables to be encoded.
        covs, [B, N, 3] torch.Tensor, covariance matrices.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diagonal: bool, if true, expects input covariances to be diagonal (full otherwise).
    Returns:
        encoded: torch.Tensor, encoded variables.
    """
    if diagonal:
        means, covs_diag = means_covs
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=means.device)  # [L]
        # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        y = rearrange(torch.unsqueeze(means, dim=-2) * torch.unsqueeze(scales, dim=-1),
                      'batch sample scale_dim mean_dim -> batch sample (scale_dim mean_dim)')
        # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        y_var = rearrange(torch.unsqueeze(covs_diag, dim=-2) * torch.unsqueeze(scales, dim=-1) ** 2,
                          'batch sample scale_dim cov_dim -> batch sample (scale_dim cov_dim)')
    else:
        means, x_cov = means_covs
        num_dims = means.shape[-1]
        # [3, L]
        basis = torch.cat([2 ** i * torch.eye(num_dims, device=means.device) for i in range(min_deg, max_deg)], 1)
        y = torch.matmul(means, basis)  # [B, N, 3] * [3, 3L] = [B, N, 3L]
        y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)
    # sin(y + 0.5 * torch.tensor(np.pi)) = cos(y)
    return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]

def pos_enc(x, min_deg, max_deg, append_identity=True):
    """The positional encoding used by the original NeRF paper."""
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device)
    # [B, 1, 3] * [L, 1] = [B, L, 3] -> [B, 3L]
    xb = rearrange(torch.unsqueeze(x, dim=-2) * torch.unsqueeze(scales, dim=-1),
                   'batch scale_dim x_dim -> batch (scale_dim x_dim)')
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.tensor(np.pi)], dim=-1))  # [B, 2*3*L]
    if append_identity:
        return torch.cat([x] + [four_feat], dim=-1)  # [B, 2*3*L+3]
    else:
        return four_feat

def volumetric_rendering(rgb, density, t_samples, dirs, white_bkgd):
    """Volumetric Rendering Function.
    Args:
        rgb: torch.Tensor, color, [batch_size, num_samples, 3]
        density: torch.Tensor, density, [batch_size, num_samples, 1].
        t_samples: torch.Tensor, [batch_size, num_samples+1].
        dirs: torch.Tensor, [batch_size, 3].
        white_bkgd: bool.
    Returns:
        comp_rgb: torch.Tensor, [batch_size, 3].
        disp: torch.Tensor, [batch_size].
        acc: torch.Tensor, [batch_size].
        weights: torch.Tensor, [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_samples[..., :-1] + t_samples[..., 1:])
    t_interval = t_samples[..., 1:] - t_samples[..., :-1]  # [B, N]
    # models/mip.py:8 here sample point by multiply the interval with the direction without normalized, so
    # the delta is norm(t1*d-t2*d) = (t1-t2)*norm(d)
    delta = t_interval * torch.linalg.norm(torch.unsqueeze(dirs, dim=-2), dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans  # [B, N]

    comp_rgb = (torch.unsqueeze(weights, dim=-1) * rgb).sum(axis=-2)  # [B, N, 1] * [B, N, 3] -> [B, 3]
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_samples[:, 0], t_samples[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - torch.unsqueeze(acc, dim=-1))
    return comp_rgb, distance, acc, weights


def rearrange_render_image(rays, chunk_size=4096):
    # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
    single_image_rays = [getattr(rays, key) for key in Rays_keys]
    val_mask = single_image_rays[-3]

    # flatten each Rays attribute and put on device
    single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # generate N Rays instances
    single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask


def contract(x):
    # x: [N, 3]
    return (2 - 1 / (torch.norm(x, dim=-1, keepdim=True))) * x / torch.norm(x, dim=-1, keepdim=True)


def parameterization(means, covs):
    '''
    means: [B, N, 3]
    covs: [B, N, 3, 3]
    '''
    contr_mask = (torch.norm(means, dim=-1) > 1).detach()
    contr_means = means[contr_mask]
    contr_covs = covs[contr_mask]
    means_clone = torch.clone(means)
    covs_clone = torch.clone(covs)
    with torch.no_grad():
        jac = vmap(jacrev(contract))(contr_means)
    contr_covs = jac @ contr_covs @ jac.permute([0, 2, 1])
    means_clone[contr_mask] = contract(contr_means)
    covs_clone[contr_mask] = contr_covs
    return means_clone, covs_clone

def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class MLP(torch.nn.Module):
    """
    A simple MLP.
    """

    def __init__(self, net_depth: int, net_width: int, net_depth_condition: int, net_width_condition: int,
                 skip_index: int, num_rgb_channels: int, num_density_channels: int, activation: str,
                 xyz_dim: int, view_dim: int):
        """
          net_depth: The depth of the first part of MLP.
          net_width: The width of the first part of MLP.
          net_depth_condition: The depth of the second part of MLP.
          net_width_condition: The width of the second part of MLP.
          activation: The activation function.
          skip_index: Add a skip connection to the output of every N layers.
          num_rgb_channels: The number of RGB channels.
          num_density_channels: The number of density channels.
        """
        super(MLP, self).__init__()
        self.skip_index: int = skip_index  # Add a skip connection to the output of every N layers.
        layers = []
        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
                dim_out = net_width
            elif (i - 1) % skip_index == 0 and i > 1:
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.layers = torch.nn.ModuleList(layers)
        del layers
        self.density_layer = torch.nn.Linear(net_width, num_density_channels)
        _xavier_init(self.density_layer)
        self.extra_layer = torch.nn.Linear(net_width, net_width)  # extra_layer is not the same as NeRF
        _xavier_init(self.extra_layer)
        layers = []
        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
                dim_out = net_width_condition
            else:
                dim_in = net_width_condition
                dim_out = net_width_condition
            linear = torch.nn.Linear(dim_in, dim_out)
            _xavier_init(linear)
            if activation == 'relu':
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            else:
                raise NotImplementedError
        self.view_layers = torch.nn.Sequential(*layers)
        del layers
        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)

    def forward(self, x, view_direction=None):
        """Evaluate the MLP.
        Args:
            x: torch.Tensor(float32), [batch, num_samples, feature], points.
            view_direction: torch.Tensor(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.
        Returns:
            raw_rgb: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_rgb_channels].
            raw_density: torch.Tensor(float32), with a shape of
                [batch, num_samples, num_density_channels].
        """
        num_samples = x.shape[1]
        inputs = x  # [B, N, 2*3*L]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % self.skip_index == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_density = self.density_layer(x)
        if view_direction is not None:
            # Output of the first part of MLP.
            bottleneck = self.extra_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            # view_direction: [B, 2*3*L] -> [B, N, 2*3*L]
            view_direction = repeat(view_direction, 'batch feature -> batch sample feature', sample=num_samples)
            x = torch.cat([bottleneck, view_direction], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            x = self.view_layers(x)
        raw_rgb = self.color_layer(x)
        return raw_rgb, raw_density

class MipNerf(torch.nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(self, N_samples_coarse: int = 128,
                 N_samples_fine: int = 128,
                 num_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_resample_grad: bool = True,
                 use_viewdirs: bool = True,
                 disparity: bool = False,
                 ray_shape: str = 'cone',
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_activation: str = 'softplus',
                 density_noise: float = 0.,
                 density_bias: float = -1.,
                 rgb_activation: str = 'sigmoid',
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,
                 append_identity: bool = True,
                 mlp_net_depth: int = 8,
                 mlp_net_width: int = 256,
                 mlp_net_depth_condition: int = 1,
                 mlp_net_width_condition: int = 128,
                 mlp_skip_index: int = 4,
                 mlp_num_rgb_channels: int = 3,
                 mlp_num_density_channels: int = 1,
                 mlp_net_activation: str = 'relu'):
        super(MipNerf, self).__init__()
        self.N_samples_coarse = N_samples_coarse  # The number of samples per level.
        self.N_samples_fine = N_samples_fine
        self.disparity = disparity  # If True, sample linearly in disparity, not in depth.
        self.ray_shape = ray_shape  # The shape of cast rays ('cone' or 'cylinder').
        self.disable_integration = disable_integration  # If True, use PE instead of IPE.
        self.min_deg_point = min_deg_point  # Min degree of positional encoding for 3D points.
        self.max_deg_point = max_deg_point  # Max degree of positional encoding for 3D points.
        self.use_viewdirs = use_viewdirs  # If True, use view directions as a condition.
        self.deg_view = deg_view  # Degree of positional encoding for viewdirs.
        self.density_noise = density_noise  # Standard deviation of noise added to raw density.
        self.density_bias = density_bias  # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding  # Dirichlet/alpha "padding" on the histogram.
        self.stop_resample_grad = stop_resample_grad  # If True, don't backprop across levels')
        mlp_xyz_dim = (max_deg_point - min_deg_point) * 3 * 2
        mlp_view_dim = deg_view * 3 * 2
        mlp_view_dim = mlp_view_dim + 3 if append_identity else mlp_view_dim
        self.mlp_coarse = MLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       mlp_xyz_dim, mlp_view_dim)
        self.mlp_fine = MLP(mlp_net_depth, mlp_net_width, mlp_net_depth_condition, mlp_net_width_condition,
                       mlp_skip_index, mlp_num_rgb_channels, mlp_num_density_channels, mlp_net_activation,
                       mlp_xyz_dim, mlp_view_dim)
        if rgb_activation == 'sigmoid':  # The RGB activation.
            self.rgb_activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
        self.rgb_padding = rgb_padding
        if density_activation == 'softplus':  # Density activation.
            self.density_activation = torch.nn.Softplus()
        else:
            raise NotImplementedError

    def forward(self, rays: namedtuple, randomized: bool, white_bkgd: bool):
        """The mip-NeRF Model.
        Args:
            rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
            randomized: bool, use randomized stratified sampling.
            white_bkgd: bool, if True, use white as the background (black o.w.).
        Returns:
            ret: list, [*(rgb, distance, acc)]
        """

        ret = []
        t_samples, weights = None, None
        t_samples, means_covs = sample_along_rays(
            rays.origins,
            rays.directions,
            rays.radii,
            self.N_samples_coarse,
            rays.near,
            rays.far,
            randomized,
            self.disparity,
            self.ray_shape,
        )
        if self.disable_integration:
            means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
        samples_enc = integrated_pos_enc(
            means_covs,
            self.min_deg_point,
            self.max_deg_point,
        )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)
        if self.use_viewdirs:
            viewdirs_enc = pos_enc(
                rays.viewdirs,
                min_deg=0,
                max_deg=self.deg_view,
                append_identity=True,
            )
            raw_rgb, raw_density = self.mlp_coarse(samples_enc, viewdirs_enc)
        else:
            raw_rgb, raw_density = self.mlp_coarse(samples_enc)
        if randomized and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

        # Volumetric rendering.
        rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
        comp_rgb, distance, acc, weights = volumetric_rendering(
            rgb,
            density,
            t_samples,
            rays.directions,
            white_bkgd=white_bkgd,
        )
        ret.append((comp_rgb, distance, acc))

        t_samples, means_covs = resample_along_rays(
            rays.origins,
            rays.directions,
            rays.radii,
            t_samples,
            weights,
            randomized,
            self.ray_shape,
            self.stop_resample_grad,
            resample_padding=self.resample_padding,
        )
        if self.disable_integration:
            means_covs = (means_covs[0], torch.zeros_like(means_covs[1]))
        samples_enc = integrated_pos_enc(
            means_covs,
            self.min_deg_point,
            self.max_deg_point,
        )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)
        if self.use_viewdirs:
            viewdirs_enc = pos_enc(
                rays.viewdirs,
                min_deg=0,
                max_deg=self.deg_view,
                append_identity=True,
            )
            raw_rgb, raw_density = self.mlp_fine(samples_enc, viewdirs_enc)
        else:
            raw_rgb, raw_density = self.mlp_fine(samples_enc)
        if randomized and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

        # Volumetric rendering.
        rgb = self.rgb_activation(raw_rgb)  # [B, N, 3]
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        density = self.density_activation(raw_density + self.density_bias)  # [B, N, 1]
        comp_rgb, distance, acc, weights = volumetric_rendering(
            rgb,
            density,
            t_samples,
            rays.directions,
            white_bkgd=white_bkgd,
        )
        ret.append((comp_rgb, distance, acc))

        return ret

class MipNerfRender(nn.Module):

    def __init__(self, config):
        super(MipNerfRender, self).__init__()

        self.config = config
        self.near_plane = config.near_plane
        self.far_plane = config.far_plane

        self.mip_nerf = MipNerf(
            N_samples_coarse=config.nerf_samples_coarse,
            N_samples_fine=config.nerf_samples_fine,
            mlp_net_depth=config.nerf_net_depth,
            mlp_net_width=config.nerf_net_width
        )
    
    def process_inputs(self, rays_o, rays_d, radii):
        near = self.near_plane * torch.ones_like(rays_o[...,:1])
        far = self.far_plane * torch.ones_like(rays_o[...,:1])
        lossmult = 0.1 * torch.ones_like(rays_o[...,:1])
        viewdirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True)+1e-4)

        rays = Rays(
            origins=rays_o,
            directions=rays_d,
            viewdirs=viewdirs,
            lossmult=lossmult,
            radii=radii,
            near=near,
            far=far
        )

        return rays
    
    def render(self, rays_o, rays_d, radii):

        rays = self.process_inputs(rays_o, rays_d, radii)
        ret = self.mip_nerf(rays, True, False)

        ret_dict = {}
        coarse_ret = ret[0]
        fine_ret = ret[1]
        ret_dict['rgb_coarse'] = coarse_ret[0]
        ret_dict['rgb_fine'] = fine_ret[0]
        ret_dict['depth_coarse'] = coarse_ret[1]
        ret_dict['depth_fine'] = fine_ret[1]

        return ret_dict
    
    def render_chunk(self, rays_o, rays_d, radii, chunk_size):
        num_rays_all = rays_o.shape[0]
        num_chunks = int(np.ceil(num_rays_all / chunk_size))

        ret_dict_list = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = (chunk_idx+1) * chunk_size
            if end_idx > num_rays_all:
                end_idx = num_rays_all
            
            rays_o_chunk = rays_o[start_idx:end_idx]
            rays_d_chunk = rays_d[start_idx:end_idx]
            radii_chunk = radii[start_idx:end_idx]
            ret_dict = self.render(rays_o_chunk, rays_d_chunk, radii_chunk)
            ret_dict_list.append(ret_dict)
        
        ret_dict = {}
        for key in ret_dict_list[0].keys():
            ret_dict[key] = torch.cat([ret_dict_list[i][key] for i in range(num_chunks)], 0)
        
        return ret_dict