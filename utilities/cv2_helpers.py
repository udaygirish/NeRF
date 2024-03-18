import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from tqdm import tqdm


def computeFundamentalMat_cv2(feats1, feats2, method="8-point"):
    """
    Computes the fundamental matrix using OpenCV's findFundamentalMat function.

    Args:
        feats1 (numpy.ndarray): Feature points from the first image.
        feats2 (numpy.ndarray): Feature points from the second image.
        method (str, optional): Method for computing the fundamental matrix. Defaults to "8-point".

    Returns:
        numpy.ndarray: The computed fundamental matrix.
    """
    if method == "8-point":
        F, _ = cv2.findFundamentalMat(feats1, feats2, cv2.FM_8POINT)
    elif method == "FM_LMEDS":
        F, _ = cv2.findFundamentalMat(feats1, feats2, cv2.FM_LMEDS)
    else:
        raise ValueError(
            "Invalid Method Selected for computation of Compute Fundamental Matrix"
        )
    return F


def computeCorrespondEpilines(points, which_image, F):
    """
    Computes the epipolar lines corresponding to the given points in the specified image.

    Args:
        points (numpy.ndarray): Array of points.
        which_image (int): Index of the image (1 or 2) to compute the epipolar lines for.
        F (numpy.ndarray): The fundamental matrix.

    Returns:
        numpy.ndarray: Array of epipolar lines.
    """
    lines = cv2.computeCorrespondEpilines(points, which_image, F)
    return lines


def compute_homography_RANSAC(pts1, pts2):
    best_inliers = []
    best_H = None
    matches = [[i, i] for i in range(len(pts1))]
    num_its = 2000
    dist_thresh = 10

    print("Running RANSAC Iterations for Homography")
    for i in tqdm(range(num_its)):
        # Randomly select 4 points
        idx = np.random.choice(len(pts1), 4, replace=False)
        rand_pts1 = pts1[idx]
        rand_pts2 = pts2[idx]

        rand_pts1 = np.float32([rand_pts1]).reshape(-1, 1, 2)
        rand_pts2 = np.float32([rand_pts2]).reshape(-1, 1, 2)

        # Compute the Perspective Transform
        H = cv2.getPerspectiveTransform(rand_pts1, rand_pts2)

        # Compute the inliers
        inliers = []
        for j, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            pt1 = np.append(pt1, 1)
            pt2 = np.append(pt2, 1)
            pt2_ = H @ pt1
            pt2_ = pt2_ / pt2_[-1]
            dist = np.linalg.norm(pt2 - pt2_)
            if dist < dist_thresh:
                inliers.append(j)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    return best_inliers


def drawlines(img1, img2, lines, pts1, pts2):
    """
    Draws the epipolar lines and corresponding points on the images.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        lines (numpy.ndarray): Array of epipolar lines.
        pts1 (numpy.ndarray): Array of points from the first image.
        pts2 (numpy.ndarray): Array of points from the second image.

    Returns:
        numpy.ndarray: The modified first image.
        numpy.ndarray: The modified second image.
    """
    if len(img1.shape) == 2:
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        r, c, _ = img1.shape
        img1 = img1
        img2 = img2
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        img1 = cv2.circle(img1, tuple(pt1), 2, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 2, color, -1)
    return img1, img2


def draw_matches_cv2(img1, img2, pts1, pts2, matches=None, matchesThickness=1):
    """
    Draws matches between two images.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        pts1 (numpy.ndarray): Array of points from the first image.
        pts2 (numpy.ndarray): Array of points from the second image.
        matchesThickness (int, optional): Thickness of the lines connecting the matches. Defaults to 1.

    Returns:
        numpy.ndarray: The combined image.
    """

    if matches is None:
        matches = [[i, i] for i in range(len(pts1))]

    # Create a new image with enough width for both images side by side
    h, w = max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1]
    combined_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Copy the first image to the left side of the combined image
    combined_img[: img1.shape[0], : img1.shape[1], :] = img1

    # Copy the second image to the right side of the combined image
    combined_img[: img2.shape[0], img1.shape[1] :, :] = img2

    # Draw lines between matches
    for match in matches:
        pt1 = tuple(map(int, pts1[match[0]]))
        pt2 = tuple(map(int, pts2[match[1]]))
        pt2 = (
            pt2[0] + img1.shape[1],
            pt2[1],
        )  # Adjust x-coordinate for the second image
        color = tuple(
            np.random.randint(0, 255, 3).tolist()
        )  # Random color for each match
        cv2.circle(
            combined_img, pt1, 2, color, -1
        )  # Draw a circle around points in the first image
        cv2.circle(
            combined_img, pt2, 2, color, -1
        )  # Draw a circle around points in the second image
        cv2.line(
            combined_img, pt1, pt2, color, matchesThickness
        )  # Draw a line between the matches

    return combined_img


def reproject_3D_2D(K, C, R, X, pts=None):
    """
    Reprojects the 3D points to the 2D Image Plane

    Args:
        K (numpy.ndarray): Camera Intrinsic Matrix
        C (numpy.ndarray): Camera Center
        R (numpy.ndarray): Rotation Matrix
        X (numpy.ndarray): 3D Points
        pts (numpy.ndarray): 2D Points

    Returns:
        numpy.ndarray: Reprojected 2D Points
    """
    I = np.identity(3)
    C = np.array(C).reshape((3, 1))

    # Projection Matrix for Camera 1
    P = K @ R @ np.hstack((I, -C))

    reprojected_2D_points = []
    for i in range(X.shape[0]):
        X_ = np.append(X[i], 1)
        X_ = X_.reshape((4, 1))
        x = P @ X_
        x = x / x[-1]
        reprojected_2D_points.append(x[:2].T)

    reprojected_2D_points = np.array(reprojected_2D_points)
    return reprojected_2D_points


def reproject_3D_2D_P(P, X):
    """
    Reprojects the 3D points to the 2D Image Plane

    Args:
        P1 (numpy.ndarray): Camera Projection Matrix
        X (numpy.ndarray): 3D Points
        pts (numpy.ndarray): 2D Points

    Returns:
        numpy.ndarray: Reprojected 2D Points
    """
    X_ = np.append(X, 1)
    X_ = X_.reshape((4, 1))
    x = P @ X_
    x = x / x[-1]
    x = x[:2].T

    x = x[0]
    return x


def project_2d_to_3d(points_2d, K, C, R):
    """
    Project 2D points to 3D using camera parameters.

    Parameters:
    - points_2d: 2D points in homogeneous coordinates (shape: (N, 3))
    - K: Camera intrinsic matrix (3x3)
    - C: Camera translation vector (3x1)
    - R: Camera rotation matrix (3x3)

    Returns:
    - points_3d: 3D points in homogeneous coordinates (shape: (N, 4))
    """
    # Invert the intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Normalize 2D points
    points_2d_normalized = K_inv.dot(points_2d.T).T

    # Compute depth using the camera translation vector
    depth = points_2d_normalized[:, 2]

    # Compute 3D points in camera coordinates
    points_3d_camera = np.hstack(
        (
            points_2d_normalized[:, :2],
            np.ones((len(points_2d_normalized), 1)),
            depth.reshape(-1, 1),
        )
    )

    # Transform 3D points to world coordinates using extrinsic parameters
    points_3d_world = R.dot(points_3d_camera.T) + C.reshape(-1, 1)

    return points_3d_world.T


def calculate_reproj_error(reproj_pts, pts):
    average_error = []
    for i in range(len(pts)):
        error = reprojection_error(reproj_pts[i][0], pts[i])
        average_error.append(error)
    return np.mean(average_error)


def reprojection_error(reprojected_pts, pts):
    """
    Computes the reprojection error.

    Args:
        reprojected_pts (numpy.ndarray): Reprojected 2D Points
        pts (numpy.ndarray): 2D Points

    Returns:
        float: Reprojection Error
    """
    reprojection_error = reprojected_pts - pts
    reprojection_error = reprojection_error[0] ** 2 + reprojection_error[1] ** 2
    # reprojection_error = np.sqrt(reprojection_error)

    return reprojection_error


def draw_points_on_image(img, pts, color=(0, 0, 255), radius=3, thickness=-1):
    """
    Draws points on the image.

    Args:
        img (numpy.ndarray): The image.
        pts (numpy.ndarray): Array of points.
        color (tuple, optional): Color of the points. Defaults to (0, 0, 255).
        radius (int, optional): Radius of the points. Defaults to 3.
        thickness (int, optional): Thickness of the points. Defaults to -1.

    Returns:
        numpy.ndarray: The modified image.
    """
    img = img.copy()
    for pt in pts:
        pt = pt[0]
        pt = tuple(map(int, pt))
        img = cv2.circle(img, tuple(pt), radius, color, thickness)
    return img


def plot_camera_pose(cam_poses, type="center", save_path="plot.png"):
    # Cam poses is list containing poses which is a tuple of (C, R)
    # Get color list based on camera poses
    color_list = ["r", "g", "b", "c", "m", "y", "k"]
    if type == "center":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i in range(len(cam_poses)):
            C, R = cam_poses[i]
            ax.scatter(C[0], C[1], C[2], c=color_list[i], marker="o")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig(save_path)
        plt.close(fig)

    elif type == "angle":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for pose in cam_poses:
            C, R = pose
            ax.quiver(
                C[0],
                C[1],
                C[2],
                R[0, 0],
                R[1, 0],
                R[2, 0],
                length=0.1,
                normalize=True,
                color="r",
            )
            ax.quiver(
                C[0],
                C[1],
                C[2],
                R[0, 1],
                R[1, 1],
                R[2, 1],
                length=0.1,
                normalize=True,
                color="g",
            )
            ax.quiver(
                C[0],
                C[1],
                C[2],
                R[0, 2],
                R[1, 2],
                R[2, 2],
                length=0.1,
                normalize=True,
                color="b",
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig(save_path)
        plt.close(fig)

    elif type == "both":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for pose in cam_poses:
            C, R = pose
            ax.scatter(C[0], C[1], C[2], c="r", marker="o")
            ax.quiver(
                C[0],
                C[1],
                C[2],
                R[0, 0],
                R[1, 0],
                R[2, 0],
                length=0.1,
                normalize=True,
                color="r",
            )
            ax.quiver(
                C[0],
                C[1],
                C[2],
                R[0, 1],
                R[1, 1],
                R[2, 1],
                length=0.1,
                normalize=True,
                color="g",
            )
            ax.quiver(
                C[0],
                C[1],
                C[2],
                R[0, 2],
                R[1, 2],
                R[2, 2],
                length=0.1,
                normalize=True,
                color="b",
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig(save_path)
        plt.close(fig)


def plot_3D_points(X_All_Poses, save_path="plot.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for i in range(len(X_All_Poses)):
        X = X_All_Poses[i]
        for j in range(len(X)):
            ax.scatter(X[j, 0], X[j, 2], c=colors[i], s=1)
    # ax.scatter(X[:, 0], X[:, 2], c=np.arange(len(X)), cmap=cm.rainbow, s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    # Equal axis scaling
    # ax.set_aspect('equal', adjustable='box')
    ax.axis("equal")
    plt.savefig(save_path)
    plt.close(fig)


def plot_3D_points_axes(X, save_path="plot.png", axis="x"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if axis == "x":
        ax.scatter(X[:, 1], X[:, 2], c=np.arange(len(X)), cmap=cm.rainbow, s=1)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
    elif axis == "y":
        ax.scatter(X[:, 0], X[:, 2], c=np.arange(len(X)), cmap=cm.rainbow, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
    elif axis == "z":
        ax.scatter(X[:, 0], X[:, 1], c=np.arange(len(X)), cmap=cm.rainbow, s=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    plt.savefig(save_path)
    plt.close(fig)


def plot_3D_points_with_Camera_poses(
    X_All_poses,
    C_set,
    R_set,
    save_path="Data/IntermediateOutputImages/Total_3d_points_Camera_plot.png",
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    for i in range(len(X_All_poses)):
        X = X_All_poses[i]
        for j in range(len(X)):
            ax.scatter(X[j, 0], X[j, 1], X[j, 2], c=colors[i], s=1)
    for i in range(len(C_set)):
        C = C_set[i]
        R = R_set[i]
        ax.scatter(C[0], C[1], C[2], c=colors[i], marker="o")
        ax.quiver(
            C[0],
            C[1],
            C[2],
            R[0, 0],
            R[1, 0],
            R[2, 0],
            length=0.1,
            normalize=True,
            color="r",
        )
        ax.quiver(
            C[0],
            C[1],
            C[2],
            R[0, 1],
            R[1, 1],
            R[2, 1],
            length=0.1,
            normalize=True,
            color="g",
        )
        ax.quiver(
            C[0],
            C[1],
            C[2],
            R[0, 2],
            R[1, 2],
            R[2, 2],
            length=0.1,
            normalize=True,
            color="b",
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(save_path)
    plt.close(fig)


def plot_2D_points_with_Camera_poses(
    X_All_poses,
    C_set,
    R_set,
    color_list,
    save_path="Data/IntermediateOutputImages/Total_2d_points_Camera_plot.png",
    legend=True,
):
    fig, ax = plt.subplots()

    for i in range(len(X_All_poses)):
        X = X_All_poses[i]
        color = color_list[i]
        # if X.shape[0] == 0:s
        #     continue
        ax.scatter(X[:, 0], X[:, 2], c=color, s=1, label=f"Points Set {i}")

    # Add origin of the world coordinate system - First camera
    marker = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
    marker._transform = marker.get_transform().rotate_deg(np.degrees(np.arctan2(0, 0)))
    ax.scatter(0, 0, c="r", marker=marker, s=100, label="1")

    for i in range(len(C_set)):
        C = C_set[i]
        R = R_set[i]
        color = color_list[i + 1]

        # Plot the camera position
        # ax.scatter(C[0], C[2], c='r', marker='o')

        # Plot the downward-facing caret marker representing camera orientation
        marker = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        marker._transform = marker.get_transform().rotate_deg(
            np.degrees(np.arctan2(R[0, 2], R[2, 2]))
        )
        ax.scatter(C[0], C[2], c=color, marker=marker, s=100, label=f"{i+1}")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.axis("equal")
    if legend:
        ax.legend()
    plt.savefig(save_path)


def plot_2D_points_with_Camera_poses_bundle(
    X_All_poses,
    C_set,
    R_set,
    color_list,
    save_path="Data/IntermediateOutputImages/Total_2d_points_Camera_plot.png",
    legend=True,
):
    fig, ax = plt.subplots()

    for i in range(len(X_All_poses)):
        X = X_All_poses[i]
        # color = color_list[i]
        if len(X) == 0:
            continue
        # if X[0] < -30 or X[0] > 20 or X[2] < -30 or X[2] > 30:
        #     continue
        ax.scatter(X[0], X[2], c="r", s=1, label=f"Points Set {i}")

    # Add origin of the world coordinate system - First camera
    marker = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
    marker._transform = marker.get_transform().rotate_deg(np.degrees(np.arctan2(0, 0)))
    ax.scatter(0, 0, c="r", marker=marker, s=100, label="1")

    for i in range(len(C_set)):
        C = C_set[i]
        R = R_set[i]
        color = color_list[i + 1]
        R = np.array(R).reshape(3, 3)

        # Plot the camera position
        # ax.scatter(C[0], C[2], c='r', marker='o')
        # Plot the downward-facing caret marker representing camera orientation
        marker = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        marker._transform = marker.get_transform().rotate_deg(
            np.degrees(np.arctan2(R[0, 2], R[2, 2]))
        )
        ax.scatter(C[0], C[2], c=color, marker=marker, s=100, label=f"{i+1}")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.axis("equal")
    if legend:
        ax.legend()
    plt.savefig(save_path)
