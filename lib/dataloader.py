import os
import sys

from sklearn import base

sys.path.append("../")
import numpy as np
import imageio
import torch
import json
import torch.nn.functional as F
import cv2
from lib.helpers import trans_t, rot_phi, rot_theta
from utilities.logger import setup_logger


class Custom_DataLoader:
    def __init__(self, base_dir_add="../nerf-synthetic/lego"):
        """
        Initializes the DataLoader object.

        Args:
            base_dir_add (str): The base directory address for the dataset. Defaults to "../nerf-synthetic/lego".

        Attributes:
            description (str): A description of the DataLoader object.
            base_dir (str): The full path of the base directory for the dataset.
            half_res (bool): A flag indicating whether to use half resolution or not.
            testskip (int): The number of frames to skip during testing.
            splits (list): A list of dataset splits, including "train", "val", and "test".
            no_of_poses (int): The number of poses in the dataset.
        """
        self.description = "DataLoader - NeRF for loading Blender Dataset based renders"
        self.base_dir = os.path.join(os.getcwd(), base_dir_add)
        self.half_res = True
        self.testskip = 8
        self.splits = ["train", "val", "test"]
        self.no_of_poses = 40

    def pose_spherical(self, theta, phi, radius):
        """
        Computes the camera-to-world transformation matrix for a given spherical pose.

        Args:
            theta (float): The azimuth angle in degrees.
            phi (float): The elevation angle in degrees.
            radius (float): The distance from the camera to the origin.

        Returns:
            torch.Tensor: The camera-to-world transformation matrix.

        """
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
        c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
        c2w = (
            torch.Tensor(
                np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            ).float()
            @ c2w
        )

        return c2w

    def load_data(self):
        """
        Loads and preprocesses the data for training or testing.

        Returns:
            all_imgs (numpy.ndarray): Array of preprocessed images.
            all_poses (numpy.ndarray): Array of pose transformation matrices.
            [H, W, focal] (list): List containing the height, width, and focal length of the images.
            count_split (list): List of indices indicating the start and end indices of each split.
        """
        data_dict = {}
        for split in self.splits:
            with open(
                os.path.join(self.base_dir, f"transforms_{split}.json"), "r"
            ) as f:
                data_dict[split] = json.load(f)

        all_imgs = []
        all_poses = []
        counts = [0]

        for i, split in enumerate(self.splits):
            data = data_dict[split]
            imgs = []
            poses = []
            if split == "train" or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip

            for frame in data["frames"][::skip]:
                fname = os.path.join(self.base_dir, frame["file_path"] + ".png")
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame["transform_matrix"]))
            imgs = (np.array(imgs) / 255.0).astype(np.float32)  # Normalize images
            poses = np.array(poses).astype(np.float32)
            all_imgs.append(imgs)
            all_poses.append(poses)
            counts.append(counts[-1] + imgs.shape[0])

        count_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
        all_imgs = np.concatenate(all_imgs, 0)
        all_poses = np.concatenate(all_poses, 0)

        H, W = all_imgs[0].shape[:2]
        camera_angle_x = float(data["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        if self.half_res:
            W = W // 2
            H = H // 2
            focal = focal / 2.0

            all_imgs_half_res = np.zeros((all_imgs.shape[0], H, W, 4), dtype=np.float32)
            for i, img in enumerate(all_imgs):
                all_imgs_half_res[i] = cv2.resize(
                    img, (W, H), interpolation=cv2.INTER_AREA
                )
            all_imgs = all_imgs_half_res

        return all_imgs, all_poses, [H, W, focal], count_split

    def render_poses(self):
        """
        Renders a set of poses using spherical coordinates.

        Returns:
            render_poses (torch.Tensor): A tensor containing the rendered poses.
        """
        render_poses = torch.stack(
            [
                self.pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, self.no_of_poses + 1)[:-1]
            ],
            0,
        )
        return render_poses


def main():
    """
    Entry point of the program.

    This function loads data using a custom data loader, logs information about the loaded data,
    and performs other necessary operations.
    """
    dataloader = Custom_DataLoader()
    all_imgs, all_poses, [H, W, focal], count_split = dataloader.load_data()
    render_poses = dataloader.render_poses()
    logger = setup_logger("DataLoader")
    logger.info("=====================================")
    logger.info("Data Loaded Successfully")
    logger.info("Image Shape: %s", all_imgs.shape)
    logger.info("Poses Shape: %s", all_poses.shape)
    logger.info("Height: %s", H)
    logger.info("Width: %s", W)
    logger.info("Focal: %s", focal)
    logger.info("Count Split: %s", count_split)
    logger.info("Render Poses: %s", render_poses.shape)
    logger.info("=====================================")


if __name__ == "__main__":
    # Run to test the dataloader
    main()
