import sys
import urllib.request
import imageio
import tensorflow as tf
import numpy as np
from itertools import chain
from model_loader import ModelLoader
import os
import pickle

def main():
    model_loader = ModelLoader("./models")
    model = model_loader.load_model("metrabs_eff2l_y4")
    print("Model loaded successfully")
    skeleton = "smpl_24"
    
    video_filepath = get_video(sys.argv[1])
    frame_batches = get_frame_batches(video_filepath, batch_size=8)
    
    # Get the first batch to determine the shape
    first_batch = next(frame_batches)
    imshape = first_batch.shape[1:3]
    
    # Create a new generator including the first batch
    frame_batches = chain([first_batch], frame_batches)
    
    # Create output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, frame_batch in enumerate(frame_batches):
        pred = model.detect_poses_batched(
            frame_batch,
            skeleton=skeleton,
            max_detections=1,
        )
        
        save_model_output(pred, i, output_dir)

def save_model_output(pred, batch_index, output_dir):
    for j, (boxes, poses) in enumerate(zip(pred["boxes"], pred["poses3d"])):
        frame_index = batch_index * 8 + j  # Assuming batch_size=8
        output_filename = f"frame_{frame_index:04d}.pkl"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "wb") as f:
            data = {"boxes": boxes.numpy(), "poses": poses.numpy()}
            pickle.dump(data, f)

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