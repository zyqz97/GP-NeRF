from argparse import Namespace
import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def get_nerf(hparams: Namespace, appearance_count: int, construct_container: bool = True) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, hparams.layer_dim, 3, 'model_state_dict', construct_container)

def get_bg_nerf(hparams: Namespace, appearance_count: int, construct_container: bool = True) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, hparams.bg_layer_dim, 4, 'bg_model_state_dict', construct_container)

def _get_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int,
                    weight_key: str, construct_container: bool = True) -> nn.Module: 
    nerf = _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim)
    if hparams.ckpt_path is not None:
        state_dict = torch.load(hparams.ckpt_path, map_location='cpu')[weight_key]
        consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')
        model_dict = nerf.state_dict()
        model_dict.update(state_dict)
        nerf.load_state_dict(model_dict)
    return nerf

def _get_single_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int) -> nn.Module:
    rgb_dim = 3 * ((hparams.sh_deg + 1) ** 2) if hparams.sh_deg is not None else 3
    
    from gp_nerf.models.gp_nerf import NeRF, ShiftedSoftplus
    return NeRF(hparams.pos_xyz_dim,
                hparams.pos_dir_dim,
                hparams.layers,
                hparams.skip_layers,
                layer_dim,
                hparams.appearance_dim,
                hparams.affine_appearance,
                appearance_count,
                rgb_dim,
                xyz_dim,
                ShiftedSoftplus() if hparams.shifted_softplus else nn.ReLU(),
                hparams)
