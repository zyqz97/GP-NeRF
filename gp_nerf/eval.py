from argparse import Namespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import sys
sys.path.append('.')

from gp_nerf.opts import get_opts_base



def _get_eval_opts() -> Namespace:
    parser = get_opts_base()

    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--centroid_path', type=str)

    return parser.parse_args()

@record
def main(hparams: Namespace) -> None:
    assert hparams.ckpt_path is not None or hparams.container_path is not None
    from gp_nerf.runner_gpnerf import Runner

    print("run clean version, remove the bg nerf")
    hparams.bg_nerf = False

    if hparams.detect_anomalies:
        with torch.autograd.detect_anomaly():
            Runner(hparams).eval()
    else:
        Runner(hparams).eval()


if __name__ == '__main__':
    main(_get_eval_opts())
