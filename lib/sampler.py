import os
import sys

sys.path.append("../")
import numpy as np
import imageio
import torch
import json
import torch.nn.functional as F
import cv2


# Hierarchial Sampling for NeRF - Probability Density Function
def hierarchial_pdf(bins, weights, N_samples, det=False):
    """
    Generate samples from a hierarchical probability density function (PDF).

    Args:
        bins (torch.Tensor): The bins of the PDF.
        weights (torch.Tensor): The weights of the PDF.
        N_samples (int): The number of samples to generate.
        det (bool, optional): Whether to use deterministic sampling.
            If True, samples are uniformly spaced. If False, samples are randomly generated.

    Returns:
        torch.Tensor: The generated samples.

    """
    weights = weights + 1e-5  # Add small value to avoid division by zero
    weights_pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(weights_pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Uniform Sampling
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Inverse CDF Sampling
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (
        cdf_g[..., 1] - cdf_g[..., 0] + 1e-5
    )  # Add small value to avoid division by zero
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
