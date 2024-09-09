# video_gen/metrabs_get_video.py
import sys
import urllib.request
import cv2
import imageio
import tensorflow as tf
import cameralib
import poseviz
import os
import numpy as np
from itertools import chain
from model_loader import ModelLoader
import pickle
import argparse
import time


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


# def load_tracked_boxes(video_filepath):
#     tracked_boxes_path = (
#         f"{os.path.splitext(os.path.basename(video_filepath))[0]}_tracked_boxes.pkl"
#     )
#     with open(tracked_boxes_path, "rb") as f:
#         return pickle.load(f)


def load_tracked_boxes(case_name, video_name):
    tracked_boxes_path = os.path.join(
        "clips", case_name, "data", f"{video_name}_tracked_boxes.pkl"
    )
    with open(tracked_boxes_path, "rb") as f:
        return pickle.load(f)


# def get_frame_batches(video_filepath, batch_size=8):
#     reader = imageio.get_reader(video_filepath)
#     frames = []
#     for frame in reader:
#         frames.append(frame)
#         if len(frames) == batch_size:
#             yield np.array(frames)
#             frames = []
#     if frames:
#         yield np.array(frames)


def load_camera_from_file(file_path, imshape):
    """
    Load camera data from a .npz file and create a Camera object using intrinsic and extrinsic matrices.

    Args:
        file_path (str): Path to the .npz file containing camera calibration data.
        imshape (tuple): Shape of the image (height, width).

    Returns:
        Camera: A Camera object initialized with the loaded data.
    """
    try:
        data = np.load(file_path)
        intrinsic_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]

        rvec = data["rvecs"][0]  # Assuming we're using the first rvec
        tvec = data["tvecs"][0]  # Assuming we're using the first tvec

        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)

        # Create 4x4 extrinsic matrix
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rot_matrix
        extrinsic_matrix[:3, 3] = tvec.flatten()

        # Set the world up vector
        world_up_vector = (0, 0, -1)

        # Create the Camera object
        camera = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            # world_up=(0, -1, 0),
            world_up=world_up_vector,
            distortion_coeffs=dist_coeffs,
        )

        return camera
    except Exception as e:
        print(f"Error loading camera data: {str(e)}")
        return None


def get_frame_batches(video_filepath, batch_size=8):
    try:
        reader = imageio.get_reader(video_filepath)
        frames = []
        for frame in reader:
            frames.append(frame)
            if len(frames) == batch_size:
                yield np.array(frames)
                frames = []
        if frames:
            yield np.array(frames)
    except Exception as e:
        print(f"Error reading video file: {str(e)}")
        raise


def update_poses_3d(poses_3d, pred, target_ids, frame_idx, batch_size):
    if pred is None:
        return

    for i, poses in enumerate(pred["poses3d"]):
        frame_num = frame_idx * batch_size + i
        for j, tid in enumerate(target_ids):
            if j < len(poses):  # Check if there's a pose for this target ID
                if tid in poses_3d:
                    poses_3d[tid][frame_num] = poses[j].numpy().reshape(1, -1, 3)


def visualize_batch(
    viz, frame_batch, batch_boxes, pred, target_ids, camera, frame_idx, batch_size
):
    if pred is None or "poses3d" not in pred:
        return

    for i, (frame, poses) in enumerate(zip(frame_batch, pred["poses3d"])):
        frame_num = frame_idx * batch_size + i
        for j, tid in enumerate(target_ids):
            if j < len(poses):
                boxes_np = np.array(batch_boxes[i][j], dtype=np.float32).reshape(1, -1)
                poses_np = poses[j].numpy().reshape(1, -1, 3)
                viz.update(frame=frame, boxes=boxes_np, poses=poses_np, camera=camera)


# def visualize_loaded_poses(
#     video_filepath,
#     poses_3d,
#     camera,
#     joint_names,
#     joint_edges,
#     target_ids,
#     output_video_path,
# ):
#     try:
#         # Print current working directory and full paths for debugging
#         print(f"Current working directory: {os.getcwd()}")
#         print(f"Input video path: {os.path.abspath(video_filepath)}")

#         # Ensure output_video_path is an absolute path
#         output_video_path = os.path.abspath(output_video_path)
#         print(f"Output video path: {output_video_path}")

#         # Ensure the output directory exists
#         output_dir = os.path.dirname(output_video_path)
#         os.makedirs(output_dir, exist_ok=True)
#         print(f"Ensured output directory exists: {output_dir}")

#         with imageio.get_reader(video_filepath) as reader:
#             fps = reader.get_meta_data()["fps"]
#             frames = [frame for frame in reader]

#         with poseviz.PoseViz(joint_names, joint_edges) as viz:
#             viz.new_sequence_output(output_video_path, fps=fps)
#             for frame_num, frame in enumerate(frames):
#                 present_target_ids = [
#                     tid
#                     for tid in target_ids
#                     if tid in poses_3d and frame_num in poses_3d[tid]
#                 ]
#                 if len(present_target_ids) > 1:
#                     print(
#                         f"Warning: More than one target ID present in frame {frame_num}: {present_target_ids}"
#                     )
#                     continue
#                 elif len(present_target_ids) == 1:
#                     tid = present_target_ids[0]
#                     poses_np = poses_3d[tid][frame_num]
#                     boxes_np = np.array(
#                         [[0, 0, frame.shape[1], frame.shape[0]]], dtype=np.float32
#                     )
#                     viz.update(
#                         frame=frame, boxes=boxes_np, poses=poses_np, camera=camera
#                     )
#     except KeyboardInterrupt:
#         print("Visualization interrupted by user.")
#     except Exception as e:
#         print(f"An error occurred during visualization: {str(e)}")
#         import traceback

#         traceback.print_exc()
#     finally:
#         print(f"Output video should be saved to {output_video_path}")


def visualize_loaded_poses_old(
    video_filepath,
    poses_3d,
    camera,
    joint_names,
    joint_edges,
    target_ids,
    output_video_path,
    add_delay=False,  # New parameter to control delay
):
    try:
        # Print current working directory and full paths for debugging
        # print(f"Current working directory: {os.getcwd()}")
        print(f"Input video path: {os.path.abspath(video_filepath)}")

        # Ensure output_video_path is an absolute path
        output_video_path = os.path.abspath(output_video_path)
        # print(f"Output video path: {output_video_path}")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_video_path)
        os.makedirs(output_dir, exist_ok=True)
        # print(f"Ensured output directory exists: {output_dir}")

        with imageio.get_reader(video_filepath) as reader:
            fps = reader.get_meta_data()["fps"]
            frames = [frame for frame in reader]

        with poseviz.PoseViz(joint_names, joint_edges, camera_view_padding=0.08) as viz:
            viz.new_sequence_output(output_video_path, fps=fps)
            for frame_num, frame in enumerate(frames):
                present_target_ids = [
                    tid
                    for tid in target_ids
                    if tid in poses_3d and frame_num in poses_3d[tid]
                ]
                if len(present_target_ids) > 1:
                    print(
                        f"Warning: More than one target ID present in frame {frame_num}: {present_target_ids}"
                    )
                    continue
                elif len(present_target_ids) == 1:
                    tid = present_target_ids[0]
                    poses_np = poses_3d[tid][frame_num]
                    boxes_np = np.array(
                        [[0, 0, frame.shape[1], frame.shape[0]]], dtype=np.float32
                    )
                    viz.update(
                        frame=frame,
                        boxes=boxes_np,
                        poses=poses_np,
                        camera=camera,
                    )

                # Add a 2-second delay between the first and second frame
                if add_delay and frame_num == 1:
                    print("Adding a 25-second delay between the first and second frame")
                    time.sleep(25)
    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
        viz.close()
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")
        # import traceback

        # traceback.print_exc()
    finally:
        print(f"Output video should be saved to {output_video_path}")
        # viz.close()


def visualize_loaded_poses(
    video_filepath,
    poses_3d,
    camera,
    joint_names,
    joint_edges,
    target_ids,
    tracked_boxes,
    output_video_path,
    add_delay=False,
):
    try:
        print(f"Input video path: {os.path.abspath(video_filepath)}")
        output_video_path = os.path.abspath(output_video_path)
        output_dir = os.path.dirname(output_video_path)
        os.makedirs(output_dir, exist_ok=True)

        with imageio.get_reader(video_filepath) as reader:
            fps = reader.get_meta_data()["fps"]
            frames = [frame for frame in reader]

        with poseviz.PoseViz(joint_names, joint_edges, camera_view_padding=0.08) as viz:
            viz.new_sequence_output(output_video_path, fps=fps)
            for frame_num, frame in enumerate(frames):
                all_poses = []
                all_boxes = []

                for tid in target_ids:
                    if tid in poses_3d and frame_num in poses_3d[tid]:
                        poses_np = poses_3d[tid][frame_num]
                        all_poses.append(poses_np)

                        # Get the bounding box from tracked_boxes
                        box = next(
                            (
                                box
                                for idx, box in tracked_boxes[tid]
                                if idx == frame_num
                            ),
                            None,
                        )
                        if box is not None:
                            ltwh_box = xywh_to_ltwh(box)
                            all_boxes.append(np.array([ltwh_box], dtype=np.float32))
                        else:
                            # If no box is found, use a default box
                            all_boxes.append(
                                np.array(
                                    [[0, 0, frame.shape[1], frame.shape[0]]],
                                    dtype=np.float32,
                                )
                            )

                if all_poses:
                    # Combine all poses and boxes
                    combined_poses = np.concatenate(all_poses, axis=0)
                    combined_boxes = np.concatenate(all_boxes, axis=0)

                    viz.update(
                        frame=frame,
                        boxes=combined_boxes,
                        poses=combined_poses,
                        camera=camera,
                    )
                else:
                    # If no poses are present, update with just the frame
                    viz.update(frame=frame, camera=camera)

                # Add a delay between the first and second frame if requested
                if add_delay and frame_num == 1:
                    print("Adding a 25-second delay between the first and second frame")
                    time.sleep(25)

    except KeyboardInterrupt:
        print("Visualization interrupted by user.")
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")
    finally:
        print(f"Output video should be saved to {output_video_path}")


def process_batch(frame_batch, tracked_boxes, target_ids, frame_idx, batch_size):
    batch_boxes = []
    present_target_ids = []
    for i in range(len(frame_batch)):
        frame_num = frame_idx * batch_size + i
        frame_boxes = []
        frame_present_target_ids = []
        for track_id in target_ids:
            if track_id in tracked_boxes:
                box = next(
                    (box for idx, box in tracked_boxes[track_id] if idx == frame_num),
                    None,
                )
                if box is not None:
                    ltwh_box = xywh_to_ltwh(box)
                    frame_boxes.append(ltwh_box)
                    frame_present_target_ids.append(track_id)
                else:
                    frame_boxes.append([0, 0, 1, 1])  # Default box
        batch_boxes.append(frame_boxes)
        present_target_ids.extend(frame_present_target_ids)

    return batch_boxes, list(set(present_target_ids))


def estimate_poses(model, frame_batch, batch_boxes, target_ids, camera, skeleton):
    if not batch_boxes:
        return None

    batch_boxes_ragged = tf.ragged.constant(
        batch_boxes, ragged_rank=1, dtype=tf.float32
    )
    # print(f"Frame batch shape: {frame_batch.shape}")
    # print(f"Batch boxes shape: {batch_boxes_ragged.shape}")
    # print(f"Camera intrinsic matrix shape: {camera.intrinsic_matrix.shape}")

    pred = model.estimate_poses_batched(
        frame_batch,
        boxes=batch_boxes_ragged,
        intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
        skeleton=skeleton,
        num_aug=20,
    )
    # print(f"Estimated model predictions = {pred}")
    return pred


def main(args):
    case_name = args.case_name
    video_name = args.video_name
    target_ids = args.target_ids
    filtered = args.filtered
    video_filepath = os.path.join("clips", case_name, "raw", f"{video_name}")
    video_name = video_name.split(".")[0]
    if filtered:
        data_path = os.path.join(
            "clips", case_name, "data", f"{video_name}_data_filtered.pkl"
        )
    else:
        data_path = os.path.join("clips", case_name, "data", f"{video_name}_data.pkl")
    output_video_path = os.path.join(
        "clips", case_name, "metrabs", f"{video_name}_output.mp4"
    )

    # Ensure we have absolute paths
    video_filepath = os.path.abspath(video_filepath)
    data_path = os.path.abspath(data_path)
    output_video_path = os.path.abspath(output_video_path)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    try:
        if args.load_poses and os.path.exists(data_path):
            print(f"Loading pre-existing data from {data_path}")
            poses_3d, camera, joint_names, joint_edges = load_data(data_path)
        else:
            print("Generating new poses using the model")

            batch_size = 12
            frame_batches = get_frame_batches(video_filepath, batch_size=batch_size)
            first_batch = next(frame_batches)
            imshape = first_batch.shape[1:3]
            frame_batches = chain([first_batch], frame_batches)

            # camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)
            # Camera initialization
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
            model = model_loader.load_model("metrabs_eff2l_y4")
            print("Model loaded successfully")

            skeleton = "smpl_24"
            joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
            joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

            tracked_boxes = load_tracked_boxes(case_name, video_name)
            poses_3d = {track_id: {} for track_id in tracked_boxes.keys()}

            viz = poseviz.PoseViz(joint_names, joint_edges) if args.visualize else None
            if viz:
                with imageio.get_reader(video_filepath) as reader:
                    fps = reader.get_meta_data()["fps"]
                viz.new_sequence_output(output_video_path, fps=fps)

            for frame_idx, frame_batch in enumerate(frame_batches):
                batch_boxes, present_target_ids = process_batch(
                    frame_batch, tracked_boxes, target_ids, frame_idx, batch_size
                )

                pred = estimate_poses(
                    model,
                    frame_batch,
                    batch_boxes,
                    present_target_ids,
                    camera,
                    skeleton,
                )
                update_poses_3d(
                    poses_3d, pred, present_target_ids, frame_idx, batch_size
                )

                if viz:
                    visualize_batch(
                        viz,
                        frame_batch,
                        batch_boxes,
                        pred,
                        present_target_ids,
                        camera,
                        frame_idx,
                        batch_size,
                    )

                print(
                    f"Processed frames {frame_idx * batch_size} to {(frame_idx + 1) * batch_size - 1}"
                )

            if viz:
                viz.close()
                print(f"Output video saved to {output_video_path}")

            save_data(data_path, poses_3d, camera, joint_names, joint_edges)

        if args.visualize and args.load_poses:
            # Make sure tracked_boxes is available here
            tracked_boxes = load_tracked_boxes(case_name, video_name)
            visualize_loaded_poses(
                video_filepath,
                poses_3d,
                camera,
                joint_names,
                joint_edges,
                target_ids,
                tracked_boxes,  # Add this line
                output_video_path,
                add_delay=True,
            )
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or load 3D poses from a video."
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
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Use filtered data file instead of unfiltered data file",
    )
    args = parser.parse_args()
    main(args)
