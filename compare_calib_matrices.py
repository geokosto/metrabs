# %%
import sys
import urllib.request
import json
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import cameralib
import poseviz
from model_loader import ModelLoader
import os
import time
from enum import Enum, auto


# %%
class CameraType(Enum):
    OLD = auto()  # old is for the json file
    NEW = auto()  # new is for the npz file


class CameraFactory:
    def __init__(
        self,
        calib_file,
        camera_type: CameraType = CameraType.OLD,
        data_path="data/camera_calibration.npz",
    ):
        self.data_path = data_path
        # with open(calib_file, "r") as file:
        #     self.calib = json.load(file)
        if camera_type == CameraType.OLD:
            with open(calib_file, "r") as file:
                self.calib = json.load(file)
        elif camera_type == CameraType.NEW:
            # calib_name = "Cam0"
            self.calib = {"Cam0": None}
            # self.calib = calib_file
        else:
            raise ValueError("Invalid camera type")
        self.camera_type = camera_type

    def create_cameras(self):
        cameras = {}
        for cam_name in self.calib:
            if self.camera_type == CameraType.OLD:
                cameras[cam_name] = self.create_camera_from_old(cam_name)
            elif self.camera_type == CameraType.NEW:
                cameras[cam_name] = self.create_camera_from_new(self.data_path)
            else:
                raise ValueError("Invalid camera type")
        return cameras

    def create_camera_from_old(self, cam_name):
        # camera_data = self.calib[cam_name]
        # extr = np.eye(4)
        # extr[:3, :3] = np.array(camera_data["rotation"])
        # extr[:3, 3] = np.array(camera_data["translation"]).flatten()
        # return cameralib.Camera(
        #     intrinsic_matrix=np.array(camera_data["intrinsicMat"]),
        #     distortion_coeffs=np.array(camera_data["distortion"]),
        #     extrinsic_matrix=extr,
        # )
        camera_data = self.calib[cam_name]
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = np.array(camera_data["R"]).reshape(3, 3)
        extrinsic_matrix[:3, 3] = np.array(camera_data["T"]).flatten()
        intrinsic_matrix = np.array(camera_data["KK"]).reshape(3, 3)
        distortion_coeffs = np.array(camera_data["kc"])

        print(f"{intrinsic_matrix=}")
        print(f"{extrinsic_matrix=}")
        print(f"{distortion_coeffs=}")
        # print(f"{world_up_vector=}")

        return cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=distortion_coeffs,
            extrinsic_matrix=extrinsic_matrix,
        )

    def create_camera_from_new(self, data_path="data/camera_calibration.npz"):
        # camera_data = self.calib[cam_name]
        # extr = np.eye(4)
        # extr[:3, :3] = np.array(camera_data["R"]).reshape(3, 3)
        # extr[:3, 3] = np.array(camera_data["T"]).flatten()
        # intrinsic_matrix = np.array(camera_data["KK"]).reshape(3, 3)
        # distortion_coeffs = np.array(camera_data["kc"])
        # return cameralib.Camera(
        #     intrinsic_matrix=intrinsic_matrix,
        #     distortion_coeffs=distortion_coeffs,
        #     extrinsic_matrix=extr,
        # )

        # Load the camera calibration data from the .npz file
        calibration_data = np.load(data_path)

        intrinsic_matrix = calibration_data["camera_matrix"]
        distortion_coeffs = calibration_data["dist_coeffs"][0]
        rvec = calibration_data["rvecs"][0]
        tvec = calibration_data["tvecs"][0]

        rot_matrix, _ = cv2.Rodrigues(rvec)

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rot_matrix
        extrinsic_matrix[:3, 3] = tvec.flatten()

        # world_up_vector = (0, 0, 1)

        print(f"{intrinsic_matrix=}")
        print(f"{extrinsic_matrix=}")
        print(f"{distortion_coeffs=}")
        # print(f"{world_up_vector=}")

        return cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=distortion_coeffs,
            extrinsic_matrix=extrinsic_matrix,
            # world_up=world_up_vector,
        )

    def get_ground_plane_height(self, extr):
        # The ground plane height is the z-coordinate of the translation vector
        return extr[2, 3]


# %%

# GROUND_TRUTH

# image_filename = "../camcourt1_1512416912203_40"
ground_truth = "./camera_params_new.json"

# Load the camera
camera_name = "Cam0"
camera_factory = CameraFactory(ground_truth, CameraType.OLD)
cameras = camera_factory.create_cameras()[camera_name]

# %%
# PREDICTION
prediction = "nikos/data/camera_calibration-camcourt1_1512416912203_40.npz"
# Load the camera
camera_name = "Cam0"
camera_factory = CameraFactory(prediction, CameraType.NEW, data_path=prediction)
cameras = camera_factory.create_cameras()[camera_name]

# %%
