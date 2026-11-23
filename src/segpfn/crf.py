"""FoRIS narrow-band refinement with the regularized Frank–Wolfe dense CRF."""

import torch
import torch.nn.functional as F

from .geometry import MEAN, SIZE, STD


class BoundaryCRF:
    def __init__(self, device):
        try:
            import CRF
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Build the dense CRF in this environment with python scripts/install_crf.py"
            ) from error
        params = CRF.FrankWolfeParams(
            scheme="fixed",
            stepsize=1.0,
            regularizer="l2",
            lambda_=1.0,
            lambda_learnable=False,
            x0_weight=0.0,
            x0_weight_learnable=False,
        )
        self.model = (
            CRF.DenseGaussianCRF(
                classes=2,
                alpha=12 * SIZE / 512,
                beta=0.03,
                gamma=4 * SIZE / 512,
                spatial_weight=3.0,
                bilateral_weight=20.0,
                compatibility=1.0,
                init="potts",
                solver="fw",
                iterations=10,
                params=params,
            )
            .to(device)
            .eval()
        )
        axis = torch.arange(-5, 6, device=device)
        self.disk = (axis[:, None] ** 2 + axis[None, :] ** 2 <= 25).float()[None, None]

    def __call__(self, image, initial):
        mask = torch.as_tensor(initial, device=image.device)
        counts = F.conv2d(mask.float()[None, None], self.disk, padding=5)[0, 0]
        core = counts == self.disk.sum()
        band = (counts > 0) ^ core
        if not band.any():
            return initial
        fg = torch.full_like(counts, 0.05)
        fg[core], fg[band] = 0.95, 0.5
        odds = torch.logit(fg)
        unary = torch.stack([-odds, odds])[None].contiguous()
        rgb = (image * image.new_tensor(STD)[:, None, None] + image.new_tensor(MEAN)[:, None, None]).clamp(
            0, 1
        )
        refined = self.model(rgb[None], unary)[0]
        return (core | (band & (refined[1] > refined[0]))).cpu().numpy()
