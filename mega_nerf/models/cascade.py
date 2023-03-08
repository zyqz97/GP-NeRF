from typing import Optional

import torch
from torch import nn


class Cascade(nn.Module):
    def __init__(self, coarse: nn.Module, fine: nn.Module):
        super(Cascade, self).__init__()
        self.coarse = coarse
        self.fine = fine

    def forward(self, use_coarse: bool, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if use_coarse:
            return self.coarse(x, sigma_only, sigma_noise)
        else:
            return self.fine(x, sigma_only, sigma_noise)


class Cascade_2model(nn.Module):
    def __init__(self, coarse: nn.Module, fine: nn.Module):
        super(Cascade_2model, self).__init__()
        self.coarse = coarse
        self.fine = fine

    def forward(self, use_coarse: bool, point_type, x: torch.Tensor, sigma_only: bool = False,
                sigma_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if use_coarse:
            return self.coarse(point_type, x, sigma_only, sigma_noise)
        else:
            return self.fine(point_type, x, sigma_only, sigma_noise)

