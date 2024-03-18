import numpy as np
import torch
import torch.nn.functional as F
import kornia
import lpips


# Helper functions for the remaining scripts


def trans_t(t):
    """
    Create a translation matrix.

    Args:
        t (float): The translation value.

    Returns:
        torch.Tensor: The translation matrix.

    """
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rot_phi(th):
    """
    Returns a 4x4 rotation matrix around the x-axis by an angle 'th'.

    Args:
        th (float): The angle of rotation in radians.

    Returns:
        torch.Tensor: A 4x4 rotation matrix.

    """
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(th), -np.sin(th), 0],
            [0, np.sin(th), np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    """
    Returns a 4x4 rotation matrix around the y-axis by the given angle.

    Args:
        th (float): The angle of rotation in radians.

    Returns:
        torch.Tensor: A 4x4 rotation matrix.

    """
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


# Losses


def mse_loss(pred, gt):
    """
    Calculates the mean squared error (MSE) loss between the predicted values and the ground truth values.

    Args:
        pred (torch.Tensor): The predicted values.
        gt (torch.Tensor): The ground truth values.

    Returns:
        torch.Tensor: The MSE loss.
    """
    return torch.mean((pred - gt) ** 2)


def psnr_loss(pred, gt):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) loss between the predicted and ground truth images.

    Parameters:
        pred (torch.Tensor): The predicted image.
        gt (torch.Tensor): The ground truth image.

    Returns:
        torch.Tensor: The PSNR loss value.
    """
    return -10 * torch.log10(mse_loss(pred, gt))


def ssim_loss(pred, gt):
    """
    Calculates the structural similarity index (SSIM) loss between the predicted image and the ground truth image.

    Args:
        pred (torch.Tensor): The predicted image.
        gt (torch.Tensor): The ground truth image.

    Returns:
        torch.Tensor: The SSIM loss between the predicted and ground truth images.
    """
    return 1- kornia.losses.ssim_loss(pred, gt, 5)


def lpips_loss(pred, gt):
    """
    Calculates the LPIPS (Learned Perceptual Image Patch Similarity) loss between predicted and ground truth images.

    Args:
        pred (torch.Tensor): Predicted images. Shape: (batch_size, channels, height, width)
        gt (torch.Tensor): Ground truth images. Shape: (batch_size, channels, height, width)

    Returns:
        torch.Tensor: Average LPIPS loss across the batch.

    """
    loss_fn_alex = lpips.LPIPS(net="alex")
    loss_fn_vgg = lpips.LPIPS(net="vgg")
    lpips_l = []
    for i in range(pred.shape[0]):
        lpips_l.append(loss_fn_vgg(pred[i], gt[i]))

    # Find the average LPIPS loss
    return torch.mean(torch.stack(lpips_l))


# Converter - to8bit


def to8bit(x):
    """
    Converts an input array to 8-bit format.

    Parameters:
    - x: numpy array
        The input array to be converted.

    Returns:
    - numpy array
        The converted array in 8-bit format.
    """
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# batchify


def batchify(fn, chunk):
    """
    Applies a function `fn` to input data in batches.

    Args:
        fn (callable): The function to be applied to each batch of input data.
        chunk (int or None): The size of each batch. If `None`, the function `fn` will be applied to the entire input data.

    Returns:
        callable: A function that applies `fn` to input data in batches.

    """
    if chunk is not None:
        return lambda x: torch.cat(
            [fn(x[i : i + chunk]) for i in range(0, x.shape[0], chunk)], 0
        )
    else:
        return fn


# run_network
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    Runs the network on the given inputs and view directions.

    Args:
        inputs (torch.Tensor): The input tensor.
        viewdirs (torch.Tensor): The view directions tensor.
        fn (callable): The network function to be applied.
        embed_fn (callable): The embedding function.
        embeddirs_fn (callable): The embedding function for view directions.
        netchunk (int, optional): The chunk size for batching the network function. Defaults to 1024 * 64.

    Returns:
        torch.Tensor: The output tensor.

    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs, list(inputs.shape[:2]) + list(outputs.shape)[1:])
    return outputs
