"""
losses.py — Differentiable loss functions for Gaussian Splatting training.

Provides SSIM loss using an 11x11 Gaussian kernel, matching the original
3D Gaussian Splatting paper (Kerbl et al., 2023).
"""

import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel [1, 1, size, size]."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Compute mean SSIM between two images.

    Args:
        img1: [B, C, H, W] predicted image
        img2: [B, C, H, W] ground truth image

    Returns:
        Scalar mean SSIM value in [0, 1].
    """
    C = img1.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma).to(img1.device, img1.dtype)
    # Expand kernel for grouped convolution across channels
    kernel = kernel.expand(C, -1, -1, -1)

    pad = window_size // 2

    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()
