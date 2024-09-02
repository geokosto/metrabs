# video_gen/metrabs_get_video.py
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


def xywh_to_ltwh(box):
    """Convert box from [x_center, y_center, width, height] to [left, top, width, height]"""
    x_center, y_center, width, height = box
    left = x_center - width / 2
    top = y_center - height / 2
    return [left, top, width, height]


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <video_filepath> <target_id>")
        sys.exit(1)

    video_filepath = sys.argv[1]
    target_id = int(sys.argv[2])  # Get the target_id from the command-line argument

    model_loader = ModelLoader("./models")
    model = model_loader.load_model("metrabs_eff2l_y4")
    print("Model loaded successfully")

    skeleton = "smpl_24"
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    frame_batch_size = 3
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

    output_video_filename = f"{video_filepath.split('/')[-1].split('.')[0]}_output.mp4"
    output_video_path = "./" + output_video_filename

    # Load tracked boxes
    tracked_boxes_path = (
        f"{video_filepath.split('/')[-1].split('.')[0]}_tracked_boxes.pkl"
    )
    with open(tracked_boxes_path, "rb") as f:
        tracked_boxes = pickle.load(f)

    # Select the boxes for the specified target_id
    target_boxes = tracked_boxes[target_id]

    # Create a dictionary to store poses for each frame
    poses_3d = {}

    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        viz.new_sequence_output(output_video_path, fps=fps)

        for frame_idx, frame_batch in enumerate(frame_batches):
            batch_boxes = []
            for i in range(len(frame_batch)):
                frame_num = frame_idx * len(frame_batch) + i
                box = next((box for idx, box in target_boxes if idx == frame_num), None)
                if box is not None:
                    # Convert box from XYWH (center-based) to LTWH (top-left based) as expected by the model
                    ltwh_box = xywh_to_ltwh(box)
                    batch_boxes.append([ltwh_box])
                else:
                    # get the last box if the current frame has no box
                    if batch_boxes:
                        batch_boxes.append(batch_boxes[-1])
                    else:
                        batch_boxes.append(
                            [[0, 0, 1, 1]]
                        )  # Default box if no detection

            # Convert batch_boxes to a RaggedTensor
            batch_boxes_ragged = tf.ragged.constant(
                batch_boxes, ragged_rank=1, inner_shape=(4,), dtype=tf.float32
            )

            pred = model.estimate_poses_batched(
                frame_batch,
                boxes=batch_boxes_ragged,
                intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
                skeleton=skeleton,
            )

            # print(f"Processing frame batch {frame_idx}")
            # print(f"Batch shape: {frame_batch.shape}")
            # print(f"Pred poses3d shape: {pred['poses3d'].shape}")

            for frame, boxes, poses in zip(frame_batch, batch_boxes, pred["poses3d"]):
                # Convert boxes and poses to numpy arrays
                boxes_np = np.array(boxes[0], dtype=np.float32)
                poses_np = poses.numpy()

                # print(f"Frame shape: {frame.shape}")
                # print(f"Boxes shape: {boxes_np.shape}")
                # print(f"Poses shape: {poses_np.shape}")

                # Ensure boxes_np has shape (N, 4) where N is the number of boxes
                if boxes_np.ndim == 1:
                    boxes_np = boxes_np.reshape(1, -1)

                # Ensure poses_np has shape (N, J, 3) where N is the number of poses and J is the number of joints
                if poses_np.ndim == 2:
                    poses_np = poses_np.reshape(1, *poses_np.shape)

                # print(f"Adjusted boxes shape: {boxes_np.shape}")
                # print(f"Adjusted poses shape: {poses_np.shape}")

                # Store the 3D poses for this frame
                poses_3d[frame_num] = poses_np

                viz.update(
                    frame=frame,  # frame is already a numpy array
                    boxes=boxes_np,
                    poses=poses_np,
                    camera=camera,
                )

    print(f"Output video saved to {output_video_path}")

    # Save the 3D poses to a pickle file
    poses_3d_filename = f"{video_filepath.split('/')[-1].split('.')[0]}_poses_3d.pkl"
    poses_3d_path = "./" + poses_3d_filename
    with open(poses_3d_path, "wb") as f:
        pickle.dump(poses_3d, f)

    print(f"3D poses saved to {poses_3d_path}")


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
    main()
