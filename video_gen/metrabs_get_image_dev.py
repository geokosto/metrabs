import sys
import os
import cv2
import imageio
import tensorflow as tf
import cameralib
import poseviz
import numpy as np
import pickle
import argparse
from model_loader import ModelLoader


def xywh_to_ltwh(box):
    x_center, y_center, width, height = box
    return [x_center - width / 2, y_center - height / 2, width, height]


def load_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data["poses_3d"], data["camera"], data["joint_names"], data["joint_edges"]


def save_data(data_path, poses_3d, camera, joint_names, joint_edges):
    with open(data_path, "wb") as f:
        pickle.dump(
            {
                "poses_3d": poses_3d,
                "camera": camera,
                "joint_names": joint_names,
                "joint_edges": joint_edges,
            },
            f,
        )
    print(f"Data saved to {data_path}")


def load_tracked_boxes(case_name, video_name):
    tracked_boxes_path = os.path.join(
        "clips", case_name, "data", f"{video_name}_tracked_boxes.pkl"
    )
    with open(tracked_boxes_path, "rb") as f:
        return pickle.load(f)


def process_frame(frame, tracked_boxes, target_ids, frame_num=0):
    frame_boxes = []
    for track_id in target_ids:
        if track_id in tracked_boxes:
            box = next(
                (box for idx, box in tracked_boxes[track_id] if idx == frame_num),
                None,
            )
            if box is not None:
                ltwh_box = xywh_to_ltwh(box)
                frame_boxes.append(ltwh_box)
            else:
                frame_boxes.append([0, 0, 1, 1])
    return frame_boxes


def load_camera_from_file_old(file_path, imshape):
    try:
        data = np.load(file_path)
        intrinsic_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        rvec = data["rvecs"][0]
        tvec = data["tvecs"][0]

        rot_matrix, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rot_matrix
        extrinsic_matrix[:3, 3] = tvec.flatten()

        world_up_vector = (0, 0, -1)

        camera = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            world_up=world_up_vector,
            distortion_coeffs=dist_coeffs,
        )
        print(f"Camera object successfully constructed.")
        return camera
    except Exception as e:
        print(f"Error loading camera data: {str(e)}")
        return None


def load_camera_from_file(file_path, imshape):
    try:
        data = np.load(file_path)
        intrinsic_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]
        rvec = data["rvecs"][0]
        tvec = data["tvecs"][0]

        rot_matrix, _ = cv2.Rodrigues(rvec)

        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rot_matrix
        extrinsic_matrix[:3, 3] = tvec.flatten()

        world_up_vector = (0, -1, 0)

        print(f"{intrinsic_matrix=}")
        print(f"{extrinsic_matrix=}")
        print(f"{dist_coeffs=}")
        print(f"{world_up_vector=}")

        camera = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            world_up=world_up_vector,
            distortion_coeffs=dist_coeffs,
        )
        print(f"Camera object successfully constructed.")
        return camera
    except Exception as e:
        print(f"Error loading camera data: {str(e)}")
        return None


def estimate_poses(model, frame, frame_boxes, target_ids, camera, skeleton):
    if not frame_boxes:
        return None
    frame_boxes_ragged = tf.ragged.constant(
        [frame_boxes], ragged_rank=1, dtype=tf.float32
    )

    pred = model.estimate_poses_batched(
        np.expand_dims(frame, axis=0),
        boxes=frame_boxes_ragged,
        intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
        extrinsic_matrix=camera.get_extrinsic_matrix()[tf.newaxis],
        distortion_coeffs=camera.distortion_coeffs,
        world_up_vector=camera.world_up,
        skeleton=skeleton,
        num_aug=3,
    )
    return pred


def visualize_single_frame(viz, frame, boxes, pred, target_ids, camera):
    if pred is None or "poses3d" not in pred:
        return
    for i, poses in enumerate(pred["poses3d"]):
        for j, tid in enumerate(target_ids):
            if j < len(poses):
                boxes_np = np.array(boxes[j], dtype=np.float32).reshape(1, -1)
                poses_np = poses[j].numpy().reshape(1, -1, 3)
                viz.update(frame=frame, boxes=boxes_np, poses=poses_np, camera=camera)


def main(args):
    case_name = args.case_name
    video_name = args.video_name
    target_ids = args.target_ids
    video_filepath = os.path.join("clips", case_name, "raw", f"{video_name}")
    video_name = video_name.split(".")[0]
    data_path = os.path.join("clips", case_name, "data", f"{video_name}_data.pkl")
    output_video_path = os.path.join(
        "clips", case_name, "metrabs", f"{video_name}_output.mp4"
    )

    try:
        if args.load_poses and os.path.exists(data_path):
            print(f"Loading pre-existing data from {data_path}")
            poses_3d, camera, joint_names, joint_edges = load_data(data_path)
        else:
            print("Generating new poses using the model")

            with imageio.get_reader(video_filepath) as reader:
                first_frame = reader.get_next_data()

            imshape = first_frame.shape[:2]
            if args.camera_data:
                camera_data_path = os.path.join(
                    "clips", case_name, "data", args.camera_data
                )
                camera = load_camera_from_file(camera_data_path, imshape)
                print(f"Camera data loaded from {camera_data_path}")
            else:
                camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)
                print("Using default camera configuration")

            model_loader = ModelLoader("./models")

            # model = model_loader.load_model("metrabs_mob3l_y4t_20211019")  # medium
            model = model_loader.load_model("metrabs_mob3s_y4t")  # smaller
            # model = model_loader.load_model("metrabs_eff2l_y4_20211019") # bigger
            print("Model loaded successfully")

            skeleton = "smpl_24"
            joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
            joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

            tracked_boxes = load_tracked_boxes(case_name, video_name)
            poses_3d = {track_id: {} for track_id in tracked_boxes.keys()}

            frame_boxes = process_frame(first_frame, tracked_boxes, target_ids)
            pred = estimate_poses(
                model, first_frame, frame_boxes, target_ids, camera, skeleton
            )
            if args.visualize:
                with poseviz.PoseViz(joint_names, joint_edges) as viz:
                    viz.new_sequence_output(output_video_path, fps=30)
                    visualize_single_frame(
                        viz, first_frame, frame_boxes, pred, target_ids, camera
                    )
                    import time

                    time.sleep(100)
                    viz.close()
                    print(f"Output video saved to {output_video_path}")

            save_data(data_path, poses_3d, camera, joint_names, joint_edges)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or load 3D poses from a single video frame."
    )
    parser.add_argument(
        "case_name", help="Name of the case directory containing the video file"
    )
    parser.add_argument(
        "video_name", help="Name of the input video file (without extension)"
    )
    parser.add_argument(
        "target_ids", type=int, nargs="+", help="Target IDs for tracking"
    )
    parser.add_argument(
        "--load_poses",
        action="store_true",
        help="Load pre-existing poses instead of generating new ones",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the poses and generate output video",
    )
    parser.add_argument(
        "--camera_data",
        help="Filename of the camera calibration data file (should be in clips/case_name/data/)",
    )
    args = parser.parse_args()
    main(args)
