import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn

#zyq : torch-ngp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

class Plane_v7(nn.Module):
    def __init__(self,hparams,
                 desired_resolution=1024,
                 base_solution=128,
                 n_levels=4,
                 ):
        super(Plane_v7, self).__init__()

        per_level_scale = np.exp2(np.log2(desired_resolution / base_solution) / (int(n_levels) - 1))
        encoding_2d_config = {
            "otype": "Grid",
            "type": "Dense",
            "n_levels": n_levels,
            "n_features_per_level": 2,
            "base_resolution": base_solution,
            "per_level_scale":per_level_scale,
        }
        self.xy = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
        self.yz = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
        self.xz = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_2d_config)
        self.feat_dim = n_levels * 2 *3

    def forward(self, x, bound):
        x = (x + bound) / (2 * bound)  # zyq: map to [0, 1]
        xy_feat = self.xy(x[:, [0, 1]])
        yz_feat = self.yz(x[:, [0, 2]])
        xz_feat = self.xz(x[:, [1, 2]])
        return torch.cat([xy_feat, yz_feat, xz_feat], dim=-1)

def get_Plane_encoder(hparams, **kwargs):
    plane_encoder = Plane_v7(hparams)
    plane_feat_dim = plane_encoder.feat_dim
    return plane_encoder, plane_feat_dim

