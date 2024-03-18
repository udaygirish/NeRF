import sys

sys.path.append("../")
import numpy as np
import torch
import torch.nn.functional as F
from lib.sampler import hierarchial_pdf
from lib.helpers import *
import torch.nn as nn

# Contains Ray Generation and Positional Encoding for NeRF


class RayGenerator:

    class DataGenerator:
        def __init__(self):
            """
            Initializes a DataGenerator object.

            Attributes:
            - description: A string representing the description of the DataGenerator.
            """
            self.description = "DataGenerator - Ray and Positional Embedding"

    def get_rays(self, H, W, K, c2w):
        """
        Compute the rays corresponding to each pixel in the image.

        Args:
            H (int): Height of the image.
            W (int): Width of the image.
            K (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
            c2w (torch.Tensor): Camera-to-world transformation matrix of shape (4, 4).

        Returns:
            tuple: A tuple containing the ray origins and ray directions.
                - ray_origins (torch.Tensor): Ray origins in the world frame of shape (H, W, 3).
                - ray_directions (torch.Tensor): Ray directions in the world frame of shape (H, W, 3).
        """
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
        i = i.t()
        j = j.t()
        directions = torch.stack(
            [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1
        )
        # Ray Directions -> Camera Frame to World Frame
        ray_directions = torch.sum(directions[..., np.newaxis, :] * c2w[:3, :3], -1)
        # Ray Origins -> Camera Frame to World Frame
        ray_origins = c2w[:3, -1].expand(ray_directions.shape)
        return ray_origins, ray_directions

    # Get rays in Numpy
    # Useful in Batching before sending as GPU Tensors or for Rendering
    def get_rays_np(self, H, W, K, c2w):
        """
        Compute the rays in the world frame given the camera parameters.

        Args:
            H (int): Height of the image.
            W (int): Width of the image.
            K (ndarray): Camera intrinsic matrix of shape (3, 3).
            c2w (ndarray): Camera-to-world transformation matrix of shape (4, 4).

        Returns:
            tuple: A tuple containing the origin and direction of the rays.
                - rays_o (ndarray): Origin of the rays in the world frame of shape (H, W, 3).
                - rays_d (ndarray): Direction of the rays in the world frame of shape (H, W, 3).
        """
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing="xy",
        )
        dirs = np.stack(
            [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
        )
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        # Translate camera frame's origin to the world frame.
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    def get_rays_ndc(self, H, W, K, near, rays_o, rays_d):
        """
        Get normalized device coordinates (NDC) for rays.

        Args:
            H (int): Height of the image.
            W (int): Width of the image.
            K (torch.Tensor): Camera intrinsic matrix.
            near (float): Distance to the near plane.
            rays_o (torch.Tensor): Ray origins.
            rays_d (torch.Tensor): Ray directions.

        Returns:
            torch.Tensor: Ray origins in NDC.
            torch.Tensor: Ray directions in NDC.
        """
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d

        # Projection of Rays to the Near Plane
        o0 = -1.0 / (W / (2.0 * K[0, 0])) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1.0 / (H / (2.0 * K[0, 0])) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1.0 + 2.0 * near / rays_o[..., 2]

        # Projection of Ray Directions to the Near Plane
        d0 = (
            -1.0
            / (W / (2.0 * K[0, 0]))
            * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
        )
        d1 = (
            -1.0
            / (H / (2.0 * K[0, 0]))
            * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
        )
        d2 = -2.0 * near / rays_o[..., 2]

        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)
        return rays_o, rays_d

    def render_rays(
        self,
        ray_batch,
        network_fn,
        network_query_fn,
        N_samples,
        include_raw=False,
        invdepth=False,
        perturb=0,
        N_importance=0,
        network_fine=None,
        white_bkgd=False,
        raw_noise_std=0.0,
    ):
        """Volumetric Rendering of Rays

        Args:
            ray_batch (torch.Tensor): Ray Batches (Shape: [batch_size, ...], Here -> Ray origin, Ray direction, min_dist, max_dist and unit magnitude direction)
            network_fn (callable): Coarse Network Function
            network_query_fn (callable): Coarse Network Query Function
            N_samples (int): Number of Samples
            include_raw (bool, optional): Include Raw unprocessed predictions. Defaults to False.
            invdepth (bool, optional): Sample with Inverse Depth if True else Sample with Linear Depth. Defaults to False.
            perturb (float, optional): Perturb Sampling if 0 else Sample with Stratified Sampling. Defaults to 0.
            N_importance (int, optional): Number of Importance Samples - Passed to Fine Network. Defaults to 0.
            network_fine (callable, optional): Fine Network Function. Defaults to None.
            white_bkgd (bool, optional): White Background if True else Black Background. Defaults to False.
            raw_noise_std (float, optional): Raw Noise Standard Deviation. Defaults to 0.0.

        Returns:
            dict: A dictionary containing the following keys:
                - "rgb_map" (torch.Tensor): Rendered RGB Map
                - "depth_map" (torch.Tensor): Rendered Depth Map
                - "acc_map" (torch.Tensor): Rendered Accumulated Map

                If include_raw is True, the dictionary will also contain the following key:
                - "raw" (torch.Tensor): Raw unprocessed predictions

                If N_importance > 0, the dictionary will also contain the following keys:
                - "rgb0" (torch.Tensor): RGB Map before importance sampling
                - "disp0" (torch.Tensor): Disparity Map before importance sampling
                - "acc0" (torch.Tensor): Accumulated Map before importance sampling
                - "z_std" (torch.Tensor): Standard deviation of the importance samples
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(
            ray_batch[..., 6:8], [-1, 1, 2]
        )  # understand more about the bounds
        near, far = bounds[..., 0], bounds[..., 1]

        # Sample Points
        t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
        if not invdepth:
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        if perturb > 0.0:
            # Stratified Sampling
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = self.rawpreds2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd
        )

        if N_importance > 0:

            # Fine network based on Importance Sampling - Hierarchial Sampling in NerF (Here the total filters - N_c + N_f (Coarse+Fine))
            rgb_map_0, disp_map_0, acc_map_0, weights_0, depth_map_0 = (
                rgb_map,
                disp_map,
                acc_map,
                weights,
                depth_map,
            )
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = hierarchial_pdf(
                z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0)
            )
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            run_fn = network_fine if network_fine is not None else network_fn
            raw = network_query_fn(pts, viewdirs, run_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = self.rawpreds2outputs(
                raw, z_vals, rays_d, raw_noise_std, white_bkgd
            )

        ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
        if include_raw:
            ret["raw"] = raw

        if N_importance > 0:
            ret["rgb0"] = rgb_map_0
            ret["disp0"] = disp_map_0
            ret["acc0"] = acc_map_0
            ret["z_std"] = torch.std(z_samples, -1)

        return ret

    def batchify_rays(
        self,
        rays_flat,
        network_fn,
        network_query_fn,
        N_samples,
        include_raw=False,
        invdepth=False,
        perturb=0,
        N_importance=0,
        network_fine=None,
        white_bkgd=False,
        raw_noise_std=0.0,
        chunk=1024 * 32,
    ):
        """
        Render rays in smaller minibatches.

        Args:
            rays_flat (torch.Tensor): Flattened rays to be rendered.
            network_fn (callable): Function that defines the network architecture.
            network_query_fn (callable): Function that queries the network for ray samples.
            N_samples (int): Number of samples per ray.
            include_raw (bool, optional): Whether to include raw RGB values. Defaults to False.
            invdepth (bool, optional): Whether to use inverse depth representation. Defaults to False.
            perturb (float, optional): Perturbation factor for ray sampling. Defaults to 0.
            N_importance (int, optional): Number of importance samples. Defaults to 0.
            network_fine (callable, optional): Fine network function. Defaults to None.
            white_bkgd (bool, optional): Whether to use white background. Defaults to False.
            raw_noise_std (float, optional): Standard deviation of raw RGB noise. Defaults to 0.0.
            chunk (int, optional): Chunk size for rendering rays in minibatches. Defaults to 1024 * 32.

        Returns:
            dict: Dictionary containing the rendered rays.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(
                rays_flat[i : i + chunk],
                network_fn,
                network_query_fn,
                N_samples,
                include_raw,
                invdepth,
                perturb,
                N_importance,
                network_fine,
                white_bkgd,
                raw_noise_std,
            )
            for k, v in ret.items():
                if i == 0:
                    all_ret[k] = []
                all_ret[k].append(v)

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def rawpreds2outputs(self, raw, z_vals, rays_d, raw_noise_std, white_bkgd):
        """Process Raw Predictions to Outputs - Important (Raw predictions are just values, here they are converted to real RGB using ReLU and Sigmoid Activation Functions)
        Args:
            raw (torch.Tensor): Raw Predictions
            z_vals (torch.Tensor): Depth Values
            rays_d (torch.Tensor): Ray Directions
            raw_noise_std (float): Raw Noise Standard Deviation
            white_bkgd (bool): White Background if True else Black Background
        Returns:
            rgb_map (torch.Tensor): Rendered RGB Map
            disp_map (torch.Tensor): Rendered Depth Map
            acc_map (torch.Tensor): Rendered Accumulated Map
            weights (torch.Tensor): Weights
            depth_map (torch.Tensor): Depth Map"""

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(
            -act_fn(raw) * dists
        )
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1
        )
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])
        noise = 0.0
        if raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        alpha = raw2alpha(raw[..., 3] + noise, dists)

        weights = (
            alpha
            * torch.cumprod(
                torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1),
                -1,
            )[:, :-1]
        )

        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map


# Positional Encoding - Embedding Class


class PositionalEncoding:
    def __init__(
        self,
        include_input=True,
        input_dims=3,
        num_freqs=10,
        max_freq_log2=9,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    ):
        """
        Initializes the DataGenerator object.

        Args:
            include_input (bool, optional): Whether to include the input in the positional encoding. Defaults to True.
            input_dims (int, optional): The number of dimensions in the input. Defaults to 3.
            num_freqs (int, optional): The number of frequencies to use in the positional encoding. Defaults to 10.
            max_freq_log2 (int, optional): The maximum frequency to use in the positional encoding. Defaults to 9.
            log_sampling (bool, optional): Whether to use logarithmic sampling for the frequencies. Defaults to True.
            periodic_fns (list, optional): The periodic functions to use in the positional encoding. Defaults to [torch.sin, torch.cos].
        """
        self.description = "Positional Encoding for NeRF"
        self.include_input = include_input
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.max_freq_log2 = self.num_freqs - 1
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        # Create the embedding function
        self.create_embedding()

    def create_embedding(self):
        """
        Creates the embedding functions and calculates the output dimensions.

        Returns:
            None
        """
        embed_functions = []
        out_dims = 0

        if self.include_input:
            embed_functions.append(lambda x: x)
            out_dims += self.input_dims

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0.0, self.max_freq_log2, steps=self.num_freqs
            )
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs
            )

        for freq in freq_bands:
            for periodic_fn in self.periodic_fns:
                embed_functions.append(
                    lambda x, periodic_fn=periodic_fn, freq=freq: periodic_fn(x * freq)
                )
                out_dims += self.input_dims

        self.embed_functions = embed_functions
        self.out_dims = out_dims

    def embed(self, inputs):
        """
        Embeds the given inputs using a set of embedding functions.

        Args:
            inputs (torch.Tensor): The input tensor to be embedded.

        Returns:
            torch.Tensor: The embedded tensor obtained by concatenating the outputs of the embedding functions.
        """
        out = torch.cat([fn(inputs) for fn in self.embed_functions], -1)
        return out


def get_embedder(multires, i=0):
    """
    Returns an embedding function and the output dimensions of the embedding.

    Parameters:
        multires (int): The number of frequency components for the positional encoding.
        i (int, optional): The index. Defaults to 0.

    Returns:
        embed (function): The embedding function.
        out_dims (int): The output dimensions of the embedding.
    """
    if i == -1:
        return nn.Identity(), 3

    embedder_obj = PositionalEncoding(num_freqs=multires)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dims
