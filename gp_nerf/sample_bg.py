import torch


def  bg_sample_inv(near, far, point_num, device):
    z = torch.linspace(0, 1, point_num, device=device)
    z_vals = 1. / near * (1 - z) + 1. / far * (z) # linear combination in the inveres space
    z_vals = 1. / z_vals # inverse back
    return z_vals


#@torch.no_grad()
def contract_to_unisphere(x: torch.Tensor, hparams):

    aabb_bound = hparams.aabb_bound
    aabb = torch.tensor([-aabb_bound, -aabb_bound, -aabb_bound, aabb_bound, aabb_bound, aabb_bound]).to(x.device)

    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    if hparams.contract_norm == 'inf':
        mag = x.abs().amax(dim=-1, keepdim=True)
    elif hparams.contract_norm == 'l2':
        mag = x.norm(dim=-1, keepdim=True)
    else:
        print("the norm of contract is wrong!")
        raise NotImplementedError
    mask = mag.squeeze(-1) > 1
    x[mask] = (1 + hparams.contract_bg_len - hparams.contract_bg_len / mag[mask]) * (x[mask] / mag[mask])  # out of bound points trun to [-2, 2]
    return x
