from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

# zyq : torch-ngp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from gp_nerf.torch_ngp.encoding import get_encoder
from gp_nerf.torch_ngp.activation import trunc_exp
from gp_nerf.models.Plane_module import get_Plane_encoder


timer = 0


class NeRF(nn.Module):
    def __init__(self, pos_xyz_dim: int,  # 12   positional embedding 
                 pos_dir_dim: int,  # 4 positional embedding 
                 layers: int,  # 8
                 skip_layers: List[int],  # [4]
                 layer_dim: int,  # 256
                 appearance_dim: int,  # 48
                 affine_appearance: bool,  # affine_appearance : False
                 appearance_count: int,  # appearance_count  : number of images (for rubble is 1678)
                 rgb_dim: int,  # rgb_dim : 3
                 xyz_dim: int,  # xyz_dim : fg = 3, bg =4
                 sigma_activation: nn.Module, hparams):
        super(NeRF, self).__init__()
        self.layer_dim = layer_dim
        print("layer_dim: {}".format(self.layer_dim))
        self.appearance_count = appearance_count
        self.appearance_dim = appearance_dim
        self.num_layers = hparams.num_layers
        self.num_layers_color = hparams.num_layers_color
        self.geo_feat_dim = hparams.geo_feat_dim
        
        #hash
        base_resolution = hparams.base_resolution
        desired_resolution = hparams.desired_resolution
        log2_hashmap_size = hparams.log2_hashmap_size
        num_levels = hparams.num_levels

        self.fg_bound = 1
        self.bg_bound = 1+hparams.contract_bg_len
        self.xyz_dim = xyz_dim

        #plane
        self.use_scaling = hparams.use_scaling
        if self.use_scaling:
            if 'quad' in hparams.dataset_path or 'sci' in hparams.dataset_path:
                self.scaling_factor_ground = (abs(hparams.sphere_center[1:]) + abs(hparams.sphere_radius[1:])) / hparams.aabb_bound
                self.scaling_factor_altitude_bottom = 0
                self.scaling_factor_altitude_range = (abs(hparams.sphere_center[0]) + abs(hparams.sphere_radius[0])) / hparams.aabb_bound
            else:
                self.scaling_factor_ground = (abs(hparams.sphere_center[1:]) + abs(hparams.sphere_radius[1:])) / hparams.aabb_bound
                self.scaling_factor_altitude_bottom = 0.5 * (hparams.z_range[0]+ hparams.z_range[1])/ hparams.aabb_bound
                self.scaling_factor_altitude_range = (hparams.z_range[1]-hparams.z_range[0]) / (2 * hparams.aabb_bound)

        self.embedding_a = nn.Embedding(self.appearance_count, self.appearance_dim)
        if 'quad' in hparams.dataset_path:
            desired_resolution_fg = desired_resolution * hparams.quad_factor
            print("Quad6k")
        else:
            desired_resolution_fg = desired_resolution
        encoding = "hashgrid"
        
        print("use two mlp")
        self.encoder, self.in_dim = get_encoder(encoding, base_resolution=base_resolution,
                                                desired_resolution=desired_resolution_fg,
                                                log2_hashmap_size=log2_hashmap_size, num_levels=num_levels)
        self.encoder_bg, _ = get_encoder(encoding, base_resolution=base_resolution,
                                            desired_resolution=desired_resolution,
                                            log2_hashmap_size=19, num_levels=num_levels)

        
        self.plane_encoder, self.plane_dim = get_Plane_encoder(hparams)
        self.sigma_net, self.color_net, self.encoder_dir = self.get_nerf_mlp()
        self.sigma_net_bg, self.color_net_bg, self.encoder_dir_bg = self.get_nerf_mlp(nerf_type='bg')

    def get_nerf_mlp(self, nerf_type='fg'):
        encoding_dir = "sphere_harmonics"
        geo_feat_dim = self.geo_feat_dim
        sigma_nets = []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.in_dim
                print("Hash and Plane")
                if nerf_type == 'fg':
                    in_dim = in_dim + self.plane_dim
                
            else:
                in_dim = self.layer_dim  # 64
            if l == self.num_layers - 1:  
                out_dim = 1 + geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = self.layer_dim
            sigma_nets.append(nn.Linear(in_dim, out_dim, bias=False))

        sigma_net = nn.ModuleList(sigma_nets)  
        encoder_dir, in_dim_dir = get_encoder(encoding_dir)
        color_nets = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = in_dim_dir + geo_feat_dim + self.appearance_dim
                if nerf_type == 'fg':
                    in_dim = in_dim + self.plane_dim
            else:
                in_dim = self.layer_dim

            if l == self.num_layers_color - 1: 
                out_dim = 3  # rgb
            else:
                out_dim = self.layer_dim

            color_nets.append(nn.Linear(in_dim, out_dim, bias=False))

        color_net = nn.ModuleList(color_nets)  
        return sigma_net, color_net, encoder_dir

    def forward(self, point_type, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:
        if point_type == 'fg':
            out = self.forward_fg(point_type, x, sigma_only, sigma_noise,train_iterations=train_iterations)
        elif point_type == 'bg':
            out = self.forward_bg(point_type, x, sigma_only, sigma_noise,train_iterations=train_iterations)
        else:
            NotImplementedError('Unkonwn point type')
        return out
    def forward_fg(self, point_type, x: torch.Tensor, sigma_only: bool = False, sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:

        position = x[:, :self.xyz_dim]
        h = self.encoder(position, bound=self.fg_bound)
        
        if self.use_scaling:
            position[:, 0] = (position[:, 0]-self.scaling_factor_altitude_bottom)/self.scaling_factor_altitude_range
            position[:, 1:] = position[:, 1:] / self.scaling_factor_ground
        plane_feat = self.plane_encoder(position, bound=self.fg_bound)
        h = torch.cat([h, plane_feat], dim=-1)

        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = x[:, self.xyz_dim:-1]
        d = self.encoder_dir(d)
        a = self.embedding_a(x[:, -1].long())
        h = torch.cat([d, geo_feat, a, plane_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        return torch.cat([color, sigma.unsqueeze(1)], -1)

    def forward_bg(self, point_type, x: torch.Tensor, sigma_only: bool = False, sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:
        position = x[:, :self.xyz_dim]
        h = self.encoder_bg(position, bound=self.bg_bound)

        for l in range(self.num_layers):
            h = self.sigma_net_bg[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = x[:, self.xyz_dim:-1]
        d = self.encoder_dir_bg(d)
        a = self.embedding_a(x[:, -1].long())
        h = torch.cat([d, geo_feat, a], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net_bg[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return torch.cat([color, sigma.unsqueeze(1)], -1)


class Embedding(nn.Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]

        return torch.cat(out, -1)


class ShiftedSoftplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

