import sys
import urllib.request

import imageio
import tensorflow as tf

# import tensorflow_hub as tfhub

import cameralib
import poseviz
import os
import numpy as np
from itertools import chain  # Import chain from itertools
from model_loader import ModelLoader  # Import the new ModelLoader class


def main():
    # model = tfhub.load("https://bit.ly/metrabs_l")
    model_loader = ModelLoader("./models")  # Create an instance of ModelLoader
    model = model_loader.load_model(
        "metrabs_eff2l_y4"
    )  # Load the model using ModelLoader
    print("Model loaded successfully")

    skeleton = "smpl_24"
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    video_filepath = get_video(sys.argv[1])

    frame_batches = get_frame_batches(video_filepath, batch_size=8)

    # Get the first batch to determine the shape
    first_batch = next(frame_batches)
    imshape = first_batch.shape[1:3]

    # Create a new generator including the first batch
    frame_batches = chain([first_batch], frame_batches)

    camera = cameralib.Camera.from_fov(fov_degrees=35, imshape=imshape)

    with imageio.get_reader(video_filepath) as reader:
        fps = reader.get_meta_data()["fps"]

    output_video_filename = f"{video_filepath.split('/')[-1].split('.')[0]}_output.mp4"
    output_video_path = "./" + output_video_filename

    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        viz.new_sequence_output(output_video_path, fps=fps)
        for frame_batch in frame_batches:
            pred = model.detect_poses_batched(
                frame_batch,
                intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
                skeleton=skeleton,
                max_detections=3,
            )

            for frame, boxes, poses in zip(frame_batch, pred["boxes"], pred["poses3d"]):
                viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)

            # save the model output to pickles for further analysis
            # save_model_output(pred, output_video_filename)


import pickle


def save_model_output(pred, output_video_filename):
    # Create a directory to save the model output
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model output to pickles
    for i, (boxes, poses) in enumerate(zip(pred["boxes"], pred["poses3d"])):
        output_filename = f"{output_video_filename.split('.')[0]}_{i}.pkl"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "wb") as f:
            data = {"boxes": boxes.numpy(), "poses": poses.numpy()}
            pickle.dump(data, f)


def load_model_output(output_video_filename):
    # Load the model output from pickles
    output_dir = "../output"
    output_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(output_video_filename.split(".")[0])
    ]

    data = []
    for output_file in output_files:
        with open(output_file, "rb") as f:
            data.append(pickle.load(f))
    return data


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
