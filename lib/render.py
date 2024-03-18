from calendar import c
import os
import sys

sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from lib.sampler import hierarchial_pdf
from lib.datagenerator import RayGenerator
from lib.helpers import *
import imageio

# Render Network for NeRF


class Renderer(RayGenerator):

    def __init__(self):
        """
        Initializes a Renderer object.

        Args:
            None

        Returns:
            None
        """
        self.description = "Renderer - NeRF"
        super(Renderer, self).__init__()

    def render(
        self,
        H,
        W,
        K,
        render_args,
        chunk=1024 * 32,
        rays=None,
        c2w=None,
        ndc=False,
        near=2.0,
        far=6.0,
        use_viewdirs=False,
        c2w_staticcam=None,
    ):
        """
        Render the NeRF model using the NeRF network.

        Args:
            H (int): Height of the image.
            W (int): Width of the image.
            K (torch.Tensor): Intrinsic camera matrix.
            render_args (list): List of rendering arguments.
            chunk (int, optional): Chunk size for batch processing. Defaults to 1024 * 32.
            rays (tuple, optional): Tuple containing the origin and direction of rays. Defaults to None.
            c2w (torch.Tensor, optional): Camera-to-world transformation matrix. Defaults to None.
            ndc (bool, optional): Whether to use NDC (Normalized Device Coordinates) for forward-facing scenes. Defaults to False.
            near (float, optional): Near clipping plane distance. Defaults to 2.0.
            far (float, optional): Far clipping plane distance. Defaults to 6.0.
            use_viewdirs (bool, optional): Whether to use view directions. Defaults to False.
            c2w_staticcam (torch.Tensor, optional): Camera-to-world transformation matrix for static camera. Defaults to None.

        Returns:
            list: List containing the rendered RGB map, disparity map, and accumulated map.
            dict: Dictionary containing additional render outputs.
        """
        if c2w is not None:
            rays_o, rays_d = self.get_rays(H, W, K, c2w)
        else:
            rays_o, rays_d = rays  # type: ignore
        if use_viewdirs:
            viewdirs = rays_d
            if c2w_staticcam is not None:
                rays_o, rays_d = self.get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        sh = rays_d.shape

        if ndc:
            # NDC used for Forward Facing scenes [-1,1]
            rays_o, rays_d = self.get_rays_ndc(H, W, K, 2.0, rays_o, rays_d)

        # Batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
            rays_d[..., :1]
        )

        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)
        # Render
        # render_args_list: [network_fn, network_query_fn, N_samples, include_raw, invdepth, perturb,
        # N_importance, network_fine, white_bkgd, raw_noise_std]
        all_ret = self.batchify_rays(
            rays,
            render_args[0],
            render_args[1],
            render_args[2],
            render_args[3],
            render_args[4],
            render_args[5],
            render_args[6],
            render_args[7],
            render_args[8],
            render_args[9],
            chunk,
        )  # type: ignore
        # print("All ret keys: ", all_ret.keys())
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ["rgb_map", "disp_map", "acc_map"]
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

        return ret_list + [ret_dict]

    def render_path(
        self,
        render_poses,
        hwf,
        K,
        chunk,
        render_args,
        gt_imgs=None,
        savedir=None,
        render_factor=0,
        use_viewdirs=True,
    ):
        """
        Render to a path of images.

        Args:
            render_poses (list): List of camera poses for rendering.
            hwf (tuple): Tuple containing the height, width, and focal length of the images.
            K (numpy.ndarray): Camera intrinsic matrix.
            chunk (int): Number of rays to process in parallel.
            render_args (dict): Additional rendering arguments.
            gt_imgs (numpy.ndarray, optional): Ground truth images for comparison.
            savedir (str, optional): Directory to save the rendered images.
            render_factor (int, optional): Downsampling factor for rendering.
            use_viewdirs (bool, optional): Whether to use view directions for rendering.

        Returns:
            rgbs (numpy.ndarray): Rendered RGB images.
            disps (numpy.ndarray): Disparity maps.

        """
        H, W, focal = hwf

        if render_factor > 0:
            H = H // render_factor
            W = W // render_factor
            focal = focal / render_factor

        rgbs = []
        disps = []

        for i, c2w in enumerate(render_poses):
            rgb, disp, acc, _ = self.render(
                H,
                W,
                K,
                render_args,
                chunk=chunk,
                c2w=c2w[:3, :4],
                use_viewdirs=use_viewdirs,
            )
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())

            if gt_imgs is not None and render_factor > 0:
                # print metrics
                print("PSNR", psnr_loss(rgbs[-1], gt_imgs))
                print("SSIM", ssim_loss(rgbs[-1], gt_imgs))

            if savedir is not None:
                rgb8 = to8bit(rgbs[-1])
                imageio.imwrite(os.path.join(savedir, f"{i:03d}.png"), rgb8)
        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

        return rgbs, disps
