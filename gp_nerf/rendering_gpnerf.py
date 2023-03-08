import os
from argparse import Namespace
from typing import Optional, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mega_nerf.spherical_harmonics import eval_sh
from gp_nerf.sample_bg import bg_sample_inv, contract_to_unisphere

TO_COMPOSITE = {'rgb', 'depth'}
INTERMEDIATE_KEYS = {'zvals_coarse', 'raw_rgb_coarse', 'raw_sigma_coarse', 'depth_real_coarse'}

def render_rays(nerf: nn.Module,
                bg_nerf: Optional[nn.Module],
                rays: torch.Tensor,
                image_indices: Optional[torch.Tensor],
                hparams: Namespace,
                sphere_center: Optional[torch.Tensor],
                sphere_radius: Optional[torch.Tensor],
                get_depth: bool,
                get_depth_variance: bool,
                get_bg_fg_rgb: bool,
                train_iterations=-1) -> Tuple[Dict[str, torch.Tensor], bool]:

    N_rays = rays.shape[0]

    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    if image_indices is not None:
        image_indices = image_indices.unsqueeze(-1).unsqueeze(-1)

    perturb = hparams.perturb if nerf.training else 0
    last_delta = 1e10 * torch.ones(N_rays, 1, device=rays.device)


    if True: #use mega's points-segmentation method with ellipsoid
        fg_far = _intersect_sphere(rays_o, rays_d, sphere_center, sphere_radius)
        fg_far = torch.maximum(fg_far, near.squeeze())
        # 划分bg ray
        rays_with_bg = torch.arange(N_rays, device=rays_o.device)[far.squeeze() > fg_far]
        rays_with_fg = torch.arange(N_rays, device=rays_o.device)[far.squeeze() <= fg_far]
        
        assert rays_with_bg.shape[0] + rays_with_fg.shape[0] == far.shape[0]
        rays_o = rays_o.view(rays_o.shape[0], 1, rays_o.shape[1])
        rays_d = rays_d.view(rays_d.shape[0], 1, rays_d.shape[1])
        if rays_with_bg.shape[0] > 0:
            last_delta[rays_with_bg, 0] = fg_far[rays_with_bg]
            # if far.max() >1:
            #     print("far.max: {}".format(far.max()))

    #  zyq:    in bound points
    far_ellipsoid = torch.minimum(far.squeeze(), fg_far).unsqueeze(-1)
    z_fg = torch.linspace(0, 1, hparams.coarse_samples, device=rays.device)
    z_vals_inbound = torch.zeros([rays_o.shape[0], hparams.coarse_samples], device=rays.device)
    z_vals_inbound[rays_with_fg] = near[rays_with_fg] * (1 - z_fg) + far_ellipsoid[rays_with_fg] * z_fg

    z_bg_inner = torch.linspace(0, 1, hparams.coarse_samples, device=rays.device)
    z_vals_inbound[rays_with_bg] = near[rays_with_bg] * (1 - z_bg_inner) + far_ellipsoid[rays_with_bg] * z_bg_inner
    z_vals_inbound = _expand_and_perturb_z_vals(z_vals_inbound, hparams.coarse_samples, perturb, N_rays)

    xyz_coarse_fg = rays_o + rays_d * z_vals_inbound.unsqueeze(-1)


    xyz_coarse_fg = contract_to_unisphere(xyz_coarse_fg, hparams)

    results = _get_results(point_type='fg',
                           nerf=nerf,
                           rays_d=rays_d,
                           image_indices=image_indices,
                           hparams=hparams,
                           xyz_coarse=xyz_coarse_fg,
                           z_vals=z_vals_inbound,
                           last_delta=last_delta,
                           get_depth=get_depth,
                           get_depth_variance=get_depth_variance,
                           get_bg_lambda=True,
                           depth_real=None,
                           xyz_fine_fn=lambda fine_z_vals: (rays_o + rays_d * fine_z_vals.unsqueeze(-1), None),
                           train_iterations=train_iterations)

    if rays_with_bg.shape[0] != 0:
        z_vals_outer = bg_sample_inv(far_ellipsoid[rays_with_bg], 1e4, hparams.coarse_samples // 2, rays.device)
        z_vals_outer = _expand_and_perturb_z_vals(z_vals_outer, hparams.coarse_samples // 2, perturb, rays_with_bg.shape[0])

        xyz_coarse_bg = rays_o[rays_with_bg] + rays_d[rays_with_bg] * z_vals_outer.unsqueeze(-1)
        xyz_coarse_bg = contract_to_unisphere(xyz_coarse_bg, hparams)
        bg_point_type='bg'
        
        bg_results = _get_results(point_type=bg_point_type,
                                  nerf=nerf,
                                  rays_d=rays_d[rays_with_bg],
                                  image_indices=image_indices[rays_with_bg] if image_indices is not None else None,
                                  hparams=hparams,
                                  xyz_coarse=xyz_coarse_bg,
                                  z_vals=z_vals_outer,
                                  # bg_nerf的last_dalta为1e10
                                  last_delta=1e10 * torch.ones(rays_with_bg.shape[0], 1, device=rays.device),
                                  get_depth=get_depth,
                                  get_depth_variance=get_depth_variance,
                                  get_bg_lambda=False,
                                  depth_real=None,
                                  xyz_fine_fn=lambda fine_z_vals: (rays_o[rays_with_bg] + rays_d[rays_with_bg] * fine_z_vals.unsqueeze(-1), None),
                                  train_iterations=train_iterations)
        
    # merge the result of inner and outer
    types = ['fine' if hparams.fine_samples > 0 else 'coarse']
    if hparams.use_cascade and hparams.fine_samples > 0:
        types.append('coarse')
    for typ in types:
        if rays_with_bg.shape[0] > 0:
            bg_lambda = results[f'bg_lambda_{typ}'][rays_with_bg]

            for key in TO_COMPOSITE:
                if f'{key}_{typ}' not in results:
                    continue

                val = results[f'{key}_{typ}']

                if get_bg_fg_rgb:
                    results[f'fg_{key}_{typ}'] = val

                expanded_bg_val = torch.zeros_like(val)

                mult = bg_lambda
                if len(val.shape) > 1:
                    mult = mult.unsqueeze(-1)

                expanded_bg_val[rays_with_bg] = bg_results[f'{key}_{typ}'] * mult

                if get_bg_fg_rgb:
                    results[f'bg_{key}_{typ}'] = expanded_bg_val

                results[f'{key}_{typ}'] = val + expanded_bg_val
        elif get_bg_fg_rgb:
            for key in TO_COMPOSITE:
                if f'{key}_{typ}' not in results:
                    continue

                val = results[f'{key}_{typ}']
                results[f'fg_{key}_{typ}'] = val
                results[f'bg_{key}_{typ}'] = torch.zeros_like(val)

    bg_nerf_rays_present = False


    return results, bg_nerf_rays_present


def _get_results(point_type,
                 nerf: nn.Module,
                 rays_d: torch.Tensor,
                 image_indices: Optional[torch.Tensor],
                 hparams: Namespace,
                 xyz_coarse: torch.Tensor,
                 z_vals: torch.Tensor,
                 last_delta: torch.Tensor,
                 get_depth: bool,
                 get_depth_variance: bool,
                 get_bg_lambda: bool,
                 depth_real: Optional[torch.Tensor],
                 xyz_fine_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]],
                 train_iterations=-1) \
        -> Dict[str, torch.Tensor]:
    results = {}

    last_delta_diff = torch.zeros_like(last_delta)
    last_delta_diff[last_delta.squeeze() < 1e10, 0] = z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]


    _inference(point_type=point_type,
               results=results,
               typ='coarse',
               nerf=nerf,
               rays_d=rays_d,
               image_indices=image_indices,
               hparams=hparams,
               xyz=xyz_coarse,
               z_vals=z_vals,
               last_delta=last_delta - last_delta_diff,
               composite_rgb=hparams.use_cascade,
               get_depth=hparams.fine_samples == 0 and get_depth,
               get_depth_variance=hparams.fine_samples == 0 and get_depth_variance,
               get_weights=hparams.fine_samples > 0,
               get_bg_lambda=get_bg_lambda and hparams.use_cascade,
               depth_real=depth_real,
               train_iterations=train_iterations)


    if hparams.fine_samples > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        perturb = hparams.perturb if nerf.training else 0
        if point_type == 'fg':
            fine_z_vals = _sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),hparams.fine_samples, det=(perturb == 0))
        elif point_type == 'bg_same_as_fg':
            fine_z_vals = _sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),hparams.fine_samples // 2, det=(perturb == 0))
        else:
            fine_z_vals = _sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),hparams.fine_samples//2, det=(perturb == 0))

        if hparams.use_cascade:
            fine_z_vals, _ = torch.sort(torch.cat([z_vals, fine_z_vals], -1), -1)

        del results['weights_coarse']


        xyz_fine, depth_real_fine = xyz_fine_fn(fine_z_vals)
        xyz_fine = contract_to_unisphere(xyz_fine, hparams)
        last_delta_diff = torch.zeros_like(last_delta)
        last_delta_diff[last_delta.squeeze() < 1e10, 0] = fine_z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]

        _inference(point_type=point_type,
                   results=results,
                   typ='fine',
                   nerf=nerf,
                   rays_d=rays_d,
                   image_indices=image_indices,
                   hparams=hparams,
                   xyz=xyz_fine,
                   z_vals=fine_z_vals,
                   last_delta=last_delta - last_delta_diff,
                   composite_rgb=True,
                   get_depth=get_depth,
                   get_depth_variance=get_depth_variance,
                   get_weights=False,
                   get_bg_lambda=get_bg_lambda,
                   depth_real=depth_real_fine,
                   train_iterations=train_iterations)

        for key in INTERMEDIATE_KEYS:
            if key in results:
                del results[key]

    return results


def _inference(point_type,
               results: Dict[str, torch.Tensor],
               typ: str,
               nerf: nn.Module,
               rays_d: torch.Tensor,
               image_indices: Optional[torch.Tensor],
               hparams: Namespace,
               xyz: torch.Tensor,
               z_vals: torch.Tensor,
               last_delta: torch.Tensor,
               composite_rgb: bool,
               get_depth: bool,
               get_depth_variance: bool,
               get_weights: bool,
               get_bg_lambda: bool,
               depth_real: Optional[torch.Tensor],
               train_iterations=-1):


    N_rays_ = xyz.shape[0]
    N_samples_ = xyz.shape[1]
    xyz_ = xyz.view(-1, xyz.shape[-1])

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []
    rays_d_ = rays_d.repeat(1, N_samples_, 1).view(-1, rays_d.shape[-1])

    if image_indices is not None:
        image_indices_ = image_indices.repeat(1, N_samples_, 1).view(-1, 1)


    # (N_rays*N_samples_, embed_dir_channels)
    for i in range(0, B, hparams.model_chunk_size):
        xyz_chunk = xyz_[i:i + hparams.model_chunk_size]

        if image_indices is not None:
            xyz_chunk = torch.cat([xyz_chunk,
                                   rays_d_[i:i + hparams.model_chunk_size],
                                   image_indices_[i:i + hparams.model_chunk_size]], 1)
        else:
            xyz_chunk = torch.cat([xyz_chunk, rays_d_[i:i + hparams.model_chunk_size]], 1)

        # sigma_noise = torch.rand(len(xyz_chunk), 1, device=xyz_chunk.device) if nerf.training else None
        sigma_noise=None

        if hparams.use_cascade:
            model_chunk = nerf(typ == 'coarse', point_type, xyz_chunk, sigma_noise=sigma_noise)
        else:
            model_chunk = nerf(point_type, xyz_chunk, sigma_noise=sigma_noise, train_iterations=train_iterations)

        out_chunks += [model_chunk]

    out = torch.cat(out_chunks, 0)
    out = out.view(N_rays_, N_samples_, out.shape[-1])

    rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples_)

    if 'zvals_coarse' in results:
        # combine coarse and fine samples
        z_vals, ordering = torch.sort(torch.cat([z_vals, results['zvals_coarse']], -1), -1, descending=False)
        rgbs = torch.cat((
            torch.gather(torch.cat((rgbs[..., 0], results['raw_rgb_coarse'][..., 0]), 1), 1, ordering).unsqueeze(
                -1),
            torch.gather(torch.cat((rgbs[..., 1], results['raw_rgb_coarse'][..., 1]), 1), 1, ordering).unsqueeze(
                -1),
            torch.gather(torch.cat((rgbs[..., 2], results['raw_rgb_coarse'][..., 2]), 1), 1, ordering).unsqueeze(-1)
        ), -1)
        sigmas = torch.gather(torch.cat((sigmas, results['raw_sigma_coarse']), 1), 1, ordering)

        if depth_real is not None:
            depth_real = torch.gather(torch.cat((depth_real, results['depth_real_coarse']), 1), 1,
                                      ordering)

    # Convert these values using volume rendering (Section 4)

    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    deltas = torch.cat([deltas, last_delta], -1)  # (N_rays, N_samples_)

    alphas = 1 - torch.exp(-deltas * sigmas)  # (N_rays, N_samples_)

    T = torch.cumprod(1 - alphas + 1e-8, -1)
    if get_bg_lambda: # only when foreground fine =True
        results[f'bg_lambda_{typ}'] = T[..., -1]

    T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]

    weights = alphas * T  # (N_rays, N_samples_)


    if get_weights: # coarse = True, fine = False
        results[f'weights_{typ}'] = weights

    if composite_rgb: # coarse = False, fine = True
        results[f'rgb_{typ}'] = (weights.unsqueeze(-1) * rgbs).sum(dim=1)  # n1 n2 c -> n1 c


    else:
        results[f'zvals_{typ}'] = z_vals
        results[f'raw_rgb_{typ}'] = rgbs
        results[f'raw_sigma_{typ}'] = sigmas
        if depth_real is not None:
            results[f'depth_real_{typ}'] = depth_real

    with torch.no_grad():
        if get_depth or get_depth_variance:
            if depth_real is not None:
                depth_map = (weights * depth_real).sum(dim=1)  # n1 n2 -> n1
            else:
                depth_map = (weights * z_vals).sum(dim=1)  # n1 n2 -> n1

        if get_depth: # always False
            results[f'depth_{typ}'] = depth_map

        if get_depth_variance:# coarse = False, fine = True
            results[f'depth_variance_{typ}'] = (weights * (z_vals - depth_map.unsqueeze(1)).square()).sum(
                axis=-1)

def _intersect_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_center: torch.Tensor,
                      sphere_radius: torch.Tensor) -> torch.Tensor:
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    '''
    rays_o, rays_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p = rays_o + d1.unsqueeze(-1) * rays_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(rays_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception(
            'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def _depth2pts_outside(rays_o: torch.Tensor, rays_d: torch.Tensor, depth: torch.Tensor, sphere_center: torch.Tensor,
                       sphere_radius: torch.Tensor, include_xyz_real: bool, cluster_2d: bool):
    '''
    rays_o, rays_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    rays_o_orig = rays_o
    rays_d_orig = rays_d
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p_mid = rays_o + d1.unsqueeze(-1) * rays_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_norm = rays_d.norm(dim=-1)
    ray_d_cos = 1. / ray_d_norm
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = rays_o + (d1 + d2).unsqueeze(-1) * rays_d

    rot_axis = torch.cross(rays_o, p_sphere, dim=-1)
    rot_axis = rot_axis / (torch.norm(rot_axis, dim=-1, keepdim=True) + 1e-8)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)

    # now calculate conventional depth
    depth_real = 1. / (depth + 1e-8) * torch.cos(theta) + d1

    if include_xyz_real:
        if cluster_2d:
            pts = torch.cat(
                (rays_o_orig + rays_d_orig * depth_real.unsqueeze(-1), p_sphere_new, depth.unsqueeze(-1)),
                dim=-1)
        else:
            boundary = rays_o_orig + rays_d_orig * (d1 + d2).unsqueeze(-1)
            pts = torch.cat((boundary.repeat(1, p_sphere_new.shape[1], 1), p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    else:
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts, depth_real


def _expand_and_perturb_z_vals(z_vals, samples, perturb, N_rays):
    z_vals = z_vals.expand(N_rays, samples)
    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    return z_vals


def _sample_pdf(bins: torch.Tensor, weights: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        fine_samples: the number of samples to draw from the distribution
        det: deterministic or not
    Outputs:
        samples: the sampled samples
    """
    weights = weights + 1e-8  # prevent division by zero (don't do inplace op!)

    pdf = weights / weights.sum(-1).unsqueeze(-1)  # (N_rays, N_samples_)

    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    return _sample_cdf(bins, cdf, fine_samples, det)


def _sample_cdf(bins: torch.Tensor, cdf: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    N_rays, N_samples_ = cdf.shape

    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive
    if det:
        u = torch.linspace(0, 1, fine_samples, device=bins.device)
        u = u.expand(N_rays, fine_samples)
    else:
        u = torch.rand(N_rays, fine_samples, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1)
    inds_sampled = inds_sampled.view(inds_sampled.shape[0], -1)  # n1 n2 2 -> n1 (n2 2)

    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(cdf_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    bins_g = torch.gather(bins, 1, inds_sampled)
    bins_g = bins_g.view(bins_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < 1e-8] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples
