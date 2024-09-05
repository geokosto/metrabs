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


def main():

    # %%

    # Set the paths
    camera_type = CameraType.OLD
    image_filename = "../camcourt1_1512416912203_40"
    calib_file = "../camera_params_new.json"
    world_up_vector = (0, 0, -1)

    # camera_type = CameraType.NEW

    # image_filename = "../camcourt1_1512416912203_40"
    # calib_file = "./data/camera_calibration-camcourt1_1512416912203_40.npz"

    # image_filename = "../sloukas"
    # calib_file = "./data/camera_calibration-sloukas.npz"

    # image_filename = "../court_left_size"
    # calib_file = "./data/camera_calibration-left_side.npz"

    # image_filename = "../full_court"
    # calib_file = "./data/camera_calibration-full_court.npz"

    # Load the image
    image_filepath = f"{image_filename}.png"
    # convert the png image to a tensor
    image = tf.image.decode_png(tf.io.read_file(image_filepath))
    # image = tf.image.decode_jpeg(tf.io.read_file(image_filepath))
    # Ensure the image is in uint8 format
    image = tf.cast(image, tf.uint8)

    # print(f"Image shape: {image.shape}")  # This will print [H, W, 3] for RGB images
    # print(f"Image dtype: {image.dtype}")  # This will print uint8
    # Convert the image from 4 channels (RGBA) to 3 channels (RGB)
    image = image[..., :3]  # Discard the alpha channel

    # Load the camera
    camera_name = "Cam0"
    camera_factory = CameraFactory(calib_file, camera_type, data_path=calib_file)
    cameras = camera_factory.create_cameras()

    # print(f"{cameras=}")

    # raise ValueError("Stop here")

    # Get ground plane height from the camera
    ground_plane_height = -camera_factory.get_ground_plane_height(
        cameras[camera_name].get_extrinsic_matrix()
    )
    print(f"Ground plane height: {ground_plane_height}")

    # %%

    # Load the model
    model_loader = ModelLoader("../models")
    model = model_loader.load_model("metrabs_mob3s_y4t")
    print("Model loaded successfully")

    # %%
    # Make the prediction
    skeleton = "smpl_24"
    # skeleton = "h36m_17"
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    # world_up_vector = (1, 0, 0)
    pred = model.detect_poses(
        image,
        detector_threshold=0.3,
        suppress_implausible_poses=True,
        max_detections=5,
        intrinsic_matrix=cameras[camera_name].intrinsic_matrix,
        extrinsic_matrix=cameras[camera_name].get_extrinsic_matrix(),
        distortion_coeffs=cameras[camera_name].distortion_coeffs,
        skeleton=skeleton,
        world_up_vector=world_up_vector,
    )
    print("Prediction successful")

    # save the prediction
    np.save(f"data/{image_filename}_pred.npy", pred)

    # %%
    # load the prediction
    pred = np.load(f"data/{image_filename}_pred.npy", allow_pickle=True).item()

    # %%

    n_views = len(cameras)
    print(f"Number of views: {n_views}")
    with poseviz.PoseViz(
        joint_names,
        joint_edges,
        n_views=n_views,
        world_up=world_up_vector,
        ground_plane_height=2 * ground_plane_height,
    ) as viz:
        frame = image.numpy()

        boxes = pred["boxes"]
        poses3d = pred["poses3d"]

        view_infos = []

        for cam_name in cameras:
            cam = cameras[cam_name]
            view_info = poseviz.ViewInfo(frame, boxes, poses3d, cam)
            view_infos.append(view_info)

        viz.update_multiview(view_infos)

        print("Press any key to exit")
        cv2.waitKey(0)

        time.sleep(1000)

    # %%``


if __name__ == "__main__":
    main()
