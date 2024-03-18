import os
from pyexpat import model
import sys
import time
from IPython import embed
from matplotlib.pyplot import step
import numpy as np
from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from lib.sampler import *
from lib.datagenerator import *
from lib.helpers import *
import imageio
from lib.render import *
from lib.sampler import hierarchial_pdf
from lib.network import *
from lib.dataloader import Custom_DataLoader
from utilities.logger import setup_logger
from utilities.utils import *
from tqdm import tqdm, trange
from torchsummary import summary

torch.set_default_tensor_type("torch.cuda.FloatTensor")
import argparse
import yaml
import shutil

# Check about config argparse ==> import configargparse
# We use Wandb (Weights and Biases for Logging)
import wandb


# global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_nerf(args):
    """
    Create a NeRF model.

    Args:
        args (dict): A dictionary containing the arguments for creating the NeRF model.

    Returns:
        tuple: A tuple containing the following elements:
            - render_args_train (list): A list of arguments for training the model.
            - render_args_test (list): A list of arguments for testing the model.
            - start (int): The starting global step for training.
            - gradient_variables (list): A list of gradient variables for optimization.
            - optimizer (torch.optim.Adam): The optimizer for the model.
    """
    # Create NeRF model
    embed_func, input_ch = get_embedder(args["multires"], args["i_embed"])
    input_ch_views = 0
    embeddirs_fn = None
    if args["use_viewdirs"]:
        embeddirs_fn, input_ch_views = get_embedder(
            args["multires_views"], args["i_embed"]
        )

    output_ch = 5 if args["N_importance"] > 0 else 4
    skips = [4]
    model = NeRF(
        W=args["num_hidden_layers"],
        D=args["network_depth"],
        input_ch=input_ch,
        input_ch_views=input_ch_views,
        output_ch=output_ch,
        skips=skips,
        use_viewdirs=args["use_viewdirs"],
    ).to(DEVICE)

    gradient_variables = list(model.parameters())
    model_fine = None
    if args["N_importance"] > 0:
        model_fine = NeRF(
            W=args["num_hidden_layers"],
            D=args["network_depth_fine"],
            input_ch=input_ch,
            input_ch_views=input_ch_views,
            output_ch=output_ch,
            skips=skips,
            use_viewdirs=args["use_viewdirs"],
        ).to(DEVICE)
        gradient_variables += list(model_fine.parameters())
        

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_func,
        embeddirs_fn=embeddirs_fn,
        netchunk=args["netchunk"],
    )

    optimizer = torch.optim.Adam(
        params=gradient_variables, lr=float(args["lrate"]), betas=(0.9, 0.999)
    )  # Need weight decay ? eps=1e-7

    start = 0
    basedir = args["basedir"]
    expname = args["expname"]

    # Load checkpoints
    if args["ckpt_path"] is not None:
        ckpt = torch.load(args["ckpt_path"], map_location=DEVICE)
        start = ckpt["global_step"]
        model.load_state_dict(ckpt["network_fn_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fn_fine_state_dict"])

    # render_args_format --> list
    #    network_fn, network_query_fn, N_samples, include_raw, invdepth, perturb,
    #    N_importance, network_fine, white_bkgd, raw_noise_std

    render_args_train = [
        model,
        network_query_fn,
        args["N_samples"],
        args["include_raw"],
        args["invdepth"],
        args["perturb"],
        args["N_importance"],
        model_fine,
        args["white_bkgd"],
        args["raw_noise_std"],
    ]

    render_args_test = [
        model,
        network_query_fn,
        args["N_samples"],
        args["include_raw"],
        args["invdepth"],
        False,  # perturb
        args["N_importance"],
        model_fine,
        args["white_bkgd"],
        0.0,
    ]

    return render_args_train, render_args_test, start, gradient_variables, optimizer


def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration.

    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for i in list(config.keys()):
        if config[i]["type"] == "int":
            config[i] = int(config[i]["default"])
        elif config[i]["type"] == "float":
            config[i] = float(config[i]["default"])
        elif config[i]["type"] == "bool":
            config[i] = bool(config[i]["default"])
        elif config[i]["type"] == "str":
            config[i] = str(config[i]["default"])
        elif config[i]["type"] == "list":
            config[i] = list(config[i]["default"])
        else:
            config[i] = config[i]["default"]
    return config


def main(default_config_path, logger):
    """
    Main function for training and rendering in the Nerf project.

    Args:
        default_config_path (str): The path to the default configuration file.
        logger: The logger object for logging messages.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path)
    args = parser.parse_args()
    # Make args such that it can be accessed as dictionary
    args = vars(args)
    config = load_config(args["config"])

    # combine the two dictionaries
    args = {**config, **args}

    if args["log_on_wandb"]:
        # Initiate wandb
        wandb.init(
            project="nerf_exp",
            config=args,
            name=args["expname"],
        )

    logger.info("=====================================")
    logger.info("Starting Training")
    logger.info("Printing Config")
    logger.info("Config: {}".format(args))
    logger.info("=====================================")
    K = None
    dataloader = Custom_DataLoader(base_dir_add=args["datadir"])
    renderer = Renderer()
    positional_encoding = PositionalEncoding(
        args["include_input"],
        args["input_ch"],
        args["multires"],
        args["multires"] - 1,
        args["log_sampling"],
    )
    ray_generator = RayGenerator()

    # Create folder to save logs and clear before
    if (
        os.path.exists(os.path.join(args["basedir"], args["expname"]))
        and args["delete_prev_results"]
    ):
        shutil.rmtree(os.path.join(args["basedir"], args["expname"]))
        os.makedirs(os.path.join(args["basedir"], args["expname"]), exist_ok=True)
    else:
        os.makedirs(os.path.join(args["basedir"], args["expname"]), exist_ok=True)

    # Load Dataset
    images, poses, hwf, count_split = dataloader.load_data()

    train, val, test = count_split
    logger.info(
        "Length of Train: {0} , Val: {1}, Test: {2}".format(
            len(train), len(val), len(test)
        )
    )
    render_poses = dataloader.render_poses()

    near = args["near"]
    far = args["far"]

    if args["white_bkgd"]:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    hwf = [int(hwf[i]) for i in [0, 1]] + [hwf[2]]

    if K is None:
        K = np.array(
            [
                [hwf[2], 0, hwf[0] // 2],
                [0, hwf[2], hwf[1] // 2],
                [0, 0, 1],
            ]
        )

    if args["render_test"]:
        render_poses = np.array(poses[test])
    # Nerf Model Creation
    render_args_train, render_args_test, start, gradient_variables, optimizer = (
        create_nerf(args)
    )

    global_step = start

    # Move data to device
    render_poses = torch.Tensor(render_poses).to(DEVICE)

    if args["render_only"]:
        with torch.no_grad():
            if args["render_test"]:
                images = images[test]
            else:
                images = None

            test_save_dir = os.path.join(
                args["basedir"],
                args["expname"],
                "renderonly_path_{:06d}".format(global_step),
            )
            os.makedirs(test_save_dir, exist_ok=True)
            rgbs, _ = renderer.render_path(
                render_poses,
                hwf,
                K,
                args["chunk"],
                render_args_train,
                images,
                savedir=test_save_dir,
                render_factor=args["render_factor"],
            )

            imageio.mimwrite(
                os.path.join(test_save_dir, "video.mp4"),
                to8bit(rgbs),
                fps=30,
                quality=8,
            )

            logger.info("Saved test set renderings to {}".format(test_save_dir))

            return

    # Training Loop
    N_rand = args["N_rand"]
    use_batching = args["use_batching"]

    if use_batching:
        rays = np.stack(
            [ray_generator.get_rays_np(hwf[0], hwf[1], K, p) for p in poses[:, :3, :4]],
            0,
        )
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in train], 0)
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)

        np.random.shuffle(rays_rgb)

        i_batch = 0

    # Move training data to device
    if use_batching:
        images = torch.Tensor(images).to(DEVICE)
        rays_rgb = torch.Tensor(rays_rgb).to(DEVICE)
    poses = torch.Tensor(poses).to(DEVICE)

    N_iters = args["N_iters"] + 1

    start = start + 1
    for i in trange(start, N_iters):
        time0 = tic()

        if use_batching:
            batch = rays_rgb[i_batch : i_batch + N_rand]
            batch = torch.transpose(batch, 0, 1)  # type: ignore
            batch_rays, target_s = batch[:2], batch[2]

            ibatch = i_batch + N_rand

            if ibatch >= rays_rgb.shape[0]:
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                ibatch = 0

        else:
            img_i = np.random.choice(train)
            target = images[img_i]
            target = torch.Tensor(target).to(DEVICE)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = ray_generator.get_rays(
                    hwf[0], hwf[1], K, torch.Tensor(pose)
                )

                if i < args["precrop_iters"]:
                    dH = int(hwf[0] // 2 * args["precrop_frac"])
                    dW = int(hwf[1] // 2 * args["precrop_frac"])
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(
                                hwf[0] // 2 - dH, hwf[0] // 2 + dH - 1, 2 * dH
                            ),
                            torch.linspace(
                                hwf[1] // 2 - dW, hwf[1] // 2 + dW - 1, 2 * dW
                            ),
                        ),
                        -1,
                    )
                    if i == start:
                        logger.info(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args['precrop_iters']}"
                        )
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, hwf[0] - 1, hwf[0]),
                            torch.linspace(0, hwf[1] - 1, hwf[1]),
                        ),
                        -1,
                    )

                coords = torch.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False
                )
                select_coords = coords[select_inds].long()
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]

        #####  Optimization Loop  ####

        # Render
        rgb, disp, acc, extras = renderer.render(
            hwf[0],
            hwf[1],
            K,
            render_args_train,
            args["chunk"],
            rays=batch_rays,
            c2w=None,
            ndc=False,
            near=2.0,
            far=6.0,
            use_viewdirs=args["use_viewdirs"],
            c2w_staticcam=None,
        )

        optimizer.zero_grad()
        img_loss = mse_loss(rgb, target_s)
        loss = img_loss
        psnr = psnr_loss(rgb, target_s)

        if "rgb0" in extras:
            imf_loss0 = mse_loss(extras["rgb0"], target_s)
            loss = loss + imf_loss0
            psnr0 = psnr_loss(extras["rgb0"], target_s)

        if args["log_on_wandb"]:
            wandb.log(
                {
                    "Train Loss": loss,
                    "Train PSNR": psnr,
                },
                step=global_step,
            )

        loss.backward()
        optimizer.step()

        # Learning rate decay - Exponential

        decay_steps = args["lrate_decay"] * args["decay_steps"]
        newlr_rate = args["lrate"] * (args["decay_rate"] ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = newlr_rate

        if args["log_on_wandb"]:
            wandb.log({"Learning Rate": newlr_rate}, step=global_step)

        if i % args["i_weights"] == 0:
            path = os.path.join(
                args["basedir"], args["expname"], "{:06d}.ckpt".format(i)
            )
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_args_train[0].state_dict(),
                    "network_fine_state_dict": render_args_train[7].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            logger.info("Saved checkpoints at {}".format(path))
            if args["log_on_wandb"]:
                wandb.log_artifact(path, type="model")

        if i % args["i_video"] == 0:
            logger.info("Making video...")

            moviebase = os.path.join(
                args["basedir"],
                args["expname"],
                "{}_spiral_{:06d}".format(args["expname"], i),
            )

            os.mkdir(moviebase)
            with torch.no_grad():
                rgbs, disps = renderer.render_path(
                    render_poses,
                    hwf,
                    K,
                    args["chunk"],
                    render_args_test,
                    savedir=moviebase,
                    use_viewdirs=args["use_viewdirs"],
                )

            logger.debug(
                "Done, saving RGBS shape:{}, DISP shape:{}".format(
                    rgbs.shape, disps.shape
                )
            )
            moviebase = moviebase + "_"
            moviebase = os.path.join(
                args["basedir"],
                args["expname"],
                "{}_spiral_{:06d}_".format(args["expname"], i),
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8bit(rgbs), fps=30, quality=8)
            # if args["log_on_wandb"]:
            #     # Log 5 images randomly
            #     for j in range(5):
            #         k = np.random.randint(0, rgbs.shape[0])
            #         wandb.log(
            #             {
            #                 "Train RGB": [
            #                     wandb.Image(to8bit(rgbs[k])),
            #                 ]
            #             }, step=global_step
            #         )

        if i % args["i_testset"] == 0:
            logger.info("Calculating Metrics on Test Set")
            render_test = test[: int(len(test) * args["image_count_test_frac"])]
            with torch.no_grad():
                rgbs, disps = renderer.render_path(
                    torch.Tensor(poses[render_test]).to(DEVICE),
                    hwf,
                    K,
                    args["chunk"],
                    render_args_test,
                    use_viewdirs=args["use_viewdirs"],
                )
            gt_imgs = images[render_test]
            gt_imgs = torch.Tensor(gt_imgs).to(DEVICE)
            rgbs = torch.Tensor(rgbs).to(DEVICE)
            temp_test_mse_loss = mse_loss(rgbs, gt_imgs)
            temp_test_psnr_loss = psnr_loss(rgbs, gt_imgs)
            temp_test_ssim_loss = ssim_loss(rgbs, gt_imgs)

            logger.info(
                "MSE Loss: {}, PSNR Loss: {}, SSIM Loss: {}".format(
                    temp_test_mse_loss, temp_test_psnr_loss, temp_test_ssim_loss
                )
            )
            rgbs_lpips = rgbs.permute(0, 3, 1, 2)
            gt_imgs_lpips = gt_imgs.permute(0, 3, 1, 2)
            # Convert from [0,255] to range [-1,1]
            rgbs_lpips = (rgbs_lpips - 127.5) / 127.5
            gt_imgs_lpips = (gt_imgs_lpips - 127.5) / 127.5
            temp_test_lpips_loss = lpips_loss(rgbs_lpips, gt_imgs_lpips)
            logger.info("LPIPS Loss: {}".format(temp_test_lpips_loss))
            if args["log_on_wandb"]:
                wandb.log(
                    {
                        "Test MSE Loss": temp_test_mse_loss,
                        "Test PSNR Loss": temp_test_psnr_loss,
                        "Test SSIM Loss": temp_test_ssim_loss,
                    },
                    step=global_step,
                )

                wandb.log(
                    {"Test LPIPS Loss": float(temp_test_lpips_loss.cpu().item()) * 1},
                    step=global_step,
                )

                # Log 2 images randomly
                for j in range(2):
                    k = np.random.randint(0, rgbs.shape[0])
                    wandb.log(
                        {
                            "Test RGB": [
                                wandb.Image(to8bit(rgbs[k].cpu().numpy())),
                            ]
                        },
                        step=global_step,
                    )
                    wandb.log(
                        {
                            "Test GT RGB": [
                                wandb.Image(to8bit(gt_imgs[k].cpu().numpy())),
                            ]
                        },
                        step=global_step,
                    )

        if i % args["iprint"] == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1


if __name__ == "__main__":

    default_config_path = "config/config.yaml"
    logger = setup_logger("umaradana_p2.log")
    main(default_config_path, logger)
