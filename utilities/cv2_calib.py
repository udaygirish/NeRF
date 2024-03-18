# General CV2 Camera Calibration Function Implementation
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from utilities.logger import setup_logger


class CamcalibCV2:

    def __init__(
        self,
        ImagesPath="Calibration_Imgs/*.jpg",
    ):
        """
        Initialize the CamcalibCV2 class.

        Args:
            ImagesPath (str): Path to the calibration images. Default is "Calibration_Imgs/*.jpg".
        """
        self.description = "Camera Calibration Class"
        # Arrays to store object points and image points
        self.objpoints = []
        self.imgpoints = []

        # Read and Make a List of Calibration images
        self.images = glob.glob(ImagesPath)

        # Prepare object points - Ex: (0,0,0), (1,0,0), (2,0,0),...
        self.objp = np.zeros((6 * 9, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    def calib(self, images_list=[]):
        """
        Perform camera calibration using the given list of images.

        Args:
            images_list (list): List of image file paths. If empty, uses the images specified during initialization.

        Returns:
            tuple: Camera matrix (mtx) and distortion coefficients (dst).
        """
        if len(images_list) == 0:
            images_list = self.images
        else:
            pass

        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        for fname in images_list:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the Chessboard Corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If corners found add object and image points

            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
            else:
                continue

            # Get Camera Matrix, distortion Coefficients, Rotational and Translation vectors,
            # for Image Undistortion

        ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None  # type: ignore
        )  # type: ignore

        self.mtx = mtx
        self.dst = dst
        self.rvecs = rvecs
        self.tvecs = tvecs

        return mtx, dst

    def undistort(self, img, mtx, dst):
        """
        Undistort the given image using the camera matrix and distortion coefficients.

        Args:
            img (numpy.ndarray): Input image.
            mtx (numpy.ndarray): Camera matrix.
            dst (numpy.ndarray): Distortion coefficients.

        Returns:
            tuple: Undistorted image, new camera matrix, and region of interest (roi).
        """
        width = img.shape[0]
        height = img.shape[1]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dst, (width, height), 0, (width, height)
        )
        undistorted_image = cv2.undistort(img, mtx, dst, None)  # , newcameramatrix)
        return undistorted_image, newcameramatrix, roi

    def calculate_projection_err(self):
        """
        Calculate the projection error.

        Returns:
            float: Total projection error.
        """
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dst
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(
                imgpoints2
            )

            mean_error += error

        mean_error = round((mean_error / len(self.objpoints)), 6)
        return mean_error
