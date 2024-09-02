import sys
import urllib.request
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


def xywh_to_ltwh(box):
    """Convert box from [x_center, y_center, width, height] to [left, top, width, height]"""
    x_center, y_center, width, height = box
    left = x_center - width / 2
    top = y_center - height / 2
    return [left, top, width, height]


def load_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data["poses_3d"], data["camera"], data["joint_names"], data["joint_edges"]


def main(args):
    video_filepath = args.video
    target_ids = args.target_ids

    output_video_filename = f"{video_filepath.split('/')[-1].split('.')[0]}_output.mp4"
    output_video_path = "./" + output_video_filename
    data_filename = f"{video_filepath.split('/')[-1].split('.')[0]}_data.pkl"
    data_path = "./" + data_filename

    if args.load_poses and os.path.exists(data_path):
        print(f"Loading pre-existing data from {data_path}")
        poses_3d, camera, joint_names, joint_edges = load_data(data_path)
    else:
        print("Generating new poses using the model")
        model_loader = ModelLoader("./models")
        model = model_loader.load_model("metrabs_eff2l_y4")
        print("Model loaded successfully")

        skeleton = "smpl_24"
        joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
        joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

        frame_batch_size = 16
        video_filepath = get_video(video_filepath)
        frame_batches = get_frame_batches(video_filepath, batch_size=frame_batch_size)

        # Get the first batch to determine the shape
        first_batch = next(frame_batches)
        imshape = first_batch.shape[1:3]

        # Construct a camera object
        fov_degrees = 55
        camera = cameralib.Camera.from_fov(fov_degrees=fov_degrees, imshape=imshape)

        # Create a new generator including the first batch
        frame_batches = chain([first_batch], frame_batches)

        with imageio.get_reader(video_filepath) as reader:
            fps = reader.get_meta_data()["fps"]

        # Load tracked boxes
        tracked_boxes_path = (
            f"{video_filepath.split('/')[-1].split('.')[0]}_tracked_boxes.pkl"
        )
        with open(tracked_boxes_path, "rb") as f:
            tracked_boxes = pickle.load(f)

        # Create a dictionary to store poses for each track ID
        poses_3d = {track_id: {} for track_id in tracked_boxes.keys()}

        viz = None
        if args.visualize:
            viz = poseviz.PoseViz(joint_names, joint_edges)
            viz.new_sequence_output(output_video_path, fps=fps)

        for frame_idx, frame_batch in enumerate(frame_batches):
            batch_boxes = {track_id: [] for track_id in tracked_boxes.keys()}
            for i in range(len(frame_batch)):
                frame_num = frame_idx * len(frame_batch) + i
                present_target_ids = []
                for track_id, boxes in tracked_boxes.items():
                    box = next((box for idx, box in boxes if idx == frame_num), None)
                    if box is not None:
                        ltwh_box = xywh_to_ltwh(box)
                        batch_boxes[track_id].append([ltwh_box])
                        if track_id in target_ids:
                            present_target_ids.append(track_id)
                    else:
                        if batch_boxes[track_id]:
                            batch_boxes[track_id].append(batch_boxes[track_id][-1])
                        else:
                            batch_boxes[track_id].append([[0, 0, 1, 1]])

                if len(present_target_ids) > 1:
                    raise ValueError(
                        f"More than one target ID present in frame {frame_num}: {present_target_ids}"
                    )

            for track_id, boxes in batch_boxes.items():
                if track_id not in target_ids:
                    continue

                batch_boxes_ragged = tf.ragged.constant(
                    boxes, ragged_rank=1, inner_shape=(4,), dtype=tf.float32
                )

                pred = model.estimate_poses_batched(
                    frame_batch,
                    boxes=batch_boxes_ragged,
                    intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
                    skeleton=skeleton,
                    # suppress_implausible_poses=True,
                    num_aug=10,
                )
                print(f"Processed frame {frame_num}")

                for i, (frame, boxes, poses) in enumerate(
                    zip(frame_batch, boxes, pred["poses3d"])
                ):
                    frame_num = frame_idx * len(frame_batch) + i
                    boxes_np = np.array(boxes[0], dtype=np.float32)
                    poses_np = poses.numpy()

                    if boxes_np.ndim == 1:
                        boxes_np = boxes_np.reshape(1, -1)
                    if poses_np.ndim == 2:
                        poses_np = poses_np.reshape(1, *poses_np.shape)

                    poses_3d[track_id][frame_num] = poses_np

                    if args.visualize and track_id in target_ids:
                        viz.update(
                            frame=frame,
                            boxes=boxes_np,
                            poses=poses_np,
                            camera=camera,
                        )

        if args.visualize:
            viz.close()
            print(f"Output video saved to {output_video_path}")

        # Save the 3D poses, camera, and other necessary data to a pickle file
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

    if args.visualize and args.load_poses:
        # Visualization using loaded poses
        with imageio.get_reader(video_filepath) as reader:
            fps = reader.get_meta_data()["fps"]
            frames = [frame for frame in reader]

        with poseviz.PoseViz(joint_names, joint_edges) as viz:
            viz.new_sequence_output(output_video_path, fps=fps)
            for frame_num, frame in enumerate(frames):
                present_target_ids = [
                    tid for tid in target_ids if frame_num in poses_3d[tid]
                ]
                if len(present_target_ids) > 1:
                    raise ValueError(
                        f"More than one target ID present in frame {frame_num}: {present_target_ids}"
                    )
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
        print(f"Output video saved to {output_video_path}")


def get_video(source, temppath="/tmp/video.mp4"):
    if not source.startswith("http"):
        return source
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath


def get_frame_batches(video_filepath, batch_size=8):
    reader = imageio.get_reader(video_filepath)
    frames = []
    for frame in reader:
        frames.append(frame)
        if len(frames) == batch_size:
            yield np.array(frames)
            frames = []
    if frames:
        yield np.array(frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or load 3D poses from a video."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--target_ids", nargs="+", type=int, help="List of target IDs for tracking"
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
    args = parser.parse_args()
    main(args)
