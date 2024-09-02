# %%
import tensorflow as tf
import os
import zipfile
import requests


def download_model(model_type):
    server_prefix = "https://omnomnom.vision.rwth-aachen.de/data/metrabs"
    model_zippath = tf.keras.utils.get_file(
        origin=f"{server_prefix}/{model_type}_20211019.zip",
        extract=True,
        cache_subdir="models",
    )
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path


def download_file(url, local_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_and_cache_model(model_type, cache_dir="models"):
    server_prefix = "https://omnomnom.vision.rwth-aachen.de/data/metrabs"
    model_filename = f"{model_type}_20211019.zip"
    cache_subdir = os.path.join(cache_dir, model_type)

    # Create cache directory if it doesn't exist
    os.makedirs(cache_subdir, exist_ok=True)

    local_zip_path = os.path.join(cache_subdir, model_filename)
    model_path = os.path.join(cache_subdir, model_type)

    # Check if the model is already downloaded and extracted
    if os.path.exists(model_path):
        print(f"Model {model_type} found in cache. Loading from {model_path}")
        return model_path

    # If not in cache, download the model
    if not os.path.exists(local_zip_path):
        print(f"Downloading {model_type} model...")
        download_file(f"{server_prefix}/{model_filename}", local_zip_path)

    # Extract the downloaded zip file
    print(f"Extracting {model_type} model...")
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(cache_subdir)

    print(f"Model {model_type} cached successfully at {model_path}")
    return model_path


def visualize_poseviz(image, pred, joint_names, joint_edges):
    # Install PoseViz from https://github.com/isarandi/poseviz
    import poseviz
    import cameralib

    camera = cameralib.Camera.from_fov(55, image.shape)
    print(camera)
    viz = poseviz.PoseViz(joint_names, joint_edges)
    viz.update(frame=image, boxes=pred["boxes"], poses=pred["poses3d"], camera=camera)


if __name__ == "__main__":
    # %%
    model_type = "metrabs_mob3l_y4t"  # or 'metrabs_eff2l_y4' for the big model
    # model_path = download_and_cache_model(model_type)
    model_path = download_model(model_type)
    model = tf.saved_model.load(model_path)

    # %%
    # model = tf.saved_model.load(
    #     download_and_extract_model("metrabs_mob3l_y4t")
    # )  # or metrabs_eff2l_y4 for the big model

    # %%
    image = tf.image.decode_jpeg(tf.io.read_file("test_image_3dpw.jpg"))
    # image = tf.image.decode_jpeg(tf.io.read_file("image.jpg"))

    # %%

    image = tf.image.decode_jpeg(tf.io.read_file("image2.jpg"))

    skeleton = "smpl_24"
    pred = model.detect_poses(image, default_fov_degrees=55, skeleton=skeleton)
    pred_nested = tf.nest.map_structure(lambda x: x.numpy(), pred)
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    visualize_poseviz(image.numpy(), pred_nested, joint_names, joint_edges)

    # %%
