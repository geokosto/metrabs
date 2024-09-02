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


def load_tracked_boxes(video_filepath):
    tracked_boxes_path = (
        f"{os.path.splitext(os.path.basename(video_filepath))[0]}_tracked_boxes.pkl"
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


def visualize_loaded_poses(
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
    )
    # print(f"Estimated model predictions = {pred}")
    return pred


def main(args):
    video_filepath = args.video
    target_ids = args.target_ids
    # data_path = f"{os.path.splitext(os.path.basename(video_filepath))[0]}_data.pkl"
    data_path = f"{os.path.splitext(os.path.basename(video_filepath))[0]}_data.pkl"
    output_video_path = (
        f"{os.path.splitext(os.path.basename(video_filepath))[0]}_output.mp4"
    )

    # Ensure we have absolute paths
    video_filepath = os.path.abspath(video_filepath)
    data_path = os.path.abspath(data_path)
    output_video_path = os.path.abspath(output_video_path)

    try:
        if args.load_poses and os.path.exists(data_path):
            print(f"Loading pre-existing data from {data_path}")
            poses_3d, camera, joint_names, joint_edges = load_data(data_path)
        else:
            print("Generating new poses using the model")
            model_loader = ModelLoader("./models")
            model = model_loader.load_model("metrabs_eff2l_y4")
            print("Model loaded successfully")

            skeleton = "smpl_24"
            # skeleton = "coco_19"
            joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
            joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

            batch_size = 12  # 8
            frame_batches = get_frame_batches(video_filepath, batch_size=batch_size)
            first_batch = next(frame_batches)
            imshape = first_batch.shape[1:3]
            camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)
            frame_batches = chain([first_batch], frame_batches)

            tracked_boxes = load_tracked_boxes(video_filepath)
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
                # print(f"{batch_boxes=}")
                # print(f"{present_target_ids=}")
                # print(f"Batch boxes shape: {len(batch_boxes)}x{len(batch_boxes[0])}x4")
                # print(f"Present target IDs: {present_target_ids}")

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
            visualize_loaded_poses(
                video_filepath,
                poses_3d,
                camera,
                joint_names,
                joint_edges,
                target_ids,
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
    parser.add_argument("video", help="Path to the input video file")
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
    args = parser.parse_args()
    main(args)
