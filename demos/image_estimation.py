import sys
import urllib.request
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import cameralib
import poseviz
from model_loader import ModelLoader
import os
import time
from enum import Enum, auto


class CameraType(Enum):
    OLD = auto()
    NEW = auto()


class CameraFactory:
    def __init__(self, calib_file, camera_type: CameraType = CameraType.OLD):
        with open(calib_file, "r") as file:
            self.calib = json.load(file)
        self.camera_type = camera_type

    def create_cameras(self):
        cameras = {}
        for cam_name in self.calib:
            if self.camera_type == CameraType.OLD:
                cameras[cam_name] = self.create_camera_from_old(cam_name)
            elif self.camera_type == CameraType.NEW:
                cameras[cam_name] = self.create_camera_from_new(cam_name)
            else:
                raise ValueError("Invalid camera type")
        return cameras

    def create_camera_from_old(self, cam_name):
        camera_data = self.calib[cam_name]
        extr = np.eye(4)
        extr[:3, :3] = np.array(camera_data["rotation"])
        extr[:3, 3] = np.array(camera_data["translation"]).flatten()
        return cameralib.Camera(
            intrinsic_matrix=np.array(camera_data["intrinsicMat"]),
            distortion_coeffs=np.array(camera_data["distortion"]),
            extrinsic_matrix=extr,
        )

    def create_camera_from_new(self, cam_name):
        camera_data = self.calib[cam_name]
        extr = np.eye(4)
        extr[:3, :3] = np.array(camera_data["R"]).reshape(3, 3)
        extr[:3, 3] = np.array(camera_data["T"]).flatten()
        intrinsic_matrix = np.array(camera_data["KK"]).reshape(3, 3)
        distortion_coeffs = np.array(camera_data["kc"])
        return cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix,
            distortion_coeffs=distortion_coeffs,
            extrinsic_matrix=extr,
        )

    def get_ground_plane_height(self, extr):
        # The ground plane height is the z-coordinate of the translation vector
        return extr[2, 3]


def main():

    # Set the paths
    models_dir = "models"
    calib_file = "camera_params_new.json"

    # Load the model
    model_loader = ModelLoader(models_dir)
    model = model_loader.load_model("metrabs_mob3l_y4t")
    print("Model loaded successfully")

    # Load the image
    image_filepath = "camcourt1_1512416912203_40.jpg"
    image = tf.image.decode_jpeg(tf.io.read_file(image_filepath))

    # Load the camera
    camera_type = CameraType.NEW
    camera_name = "Cam0"
    camera_factory = CameraFactory(calib_file, camera_type)
    cameras = camera_factory.create_cameras()

    # Get ground plane height from the camera
    ground_plane_height = -camera_factory.get_ground_plane_height(
        cameras[camera_name].extrinsic_matrix
    )
    print(f"Ground plane height: {ground_plane_height}")

    # Make the prediction
    skeleton = "smpl_24"
    # skeleton = "h36m_17"
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    world_up_vector = (0, 0, -1)
    pred = model.detect_poses(
        image,
        detector_threshold=0.3,
        suppress_implausible_poses=False,
        max_detections=5,
        intrinsic_matrix=cameras[camera_name].intrinsic_matrix,
        extrinsic_matrix=cameras[camera_name].extrinsic_matrix,
        distortion_coeffs=cameras[camera_name].distortion_coeffs,
        skeleton=skeleton,
        world_up_vector=world_up_vector,
    )
    print("Prediction successful")

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

        time.sleep(1000)


if __name__ == "__main__":
    main()
