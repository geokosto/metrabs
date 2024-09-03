# %%
# smooth_markers.py
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle

# load the metrabs pkl data
case_name = "pao_promo"
video_name = "lessort-dunk-01"
datapath = f"clips/{case_name}/data/{video_name}_data.pkl"
with open(datapath, "rb") as f:
    data = pd.read_pickle(f)

# %%
# Set track id
track_id = 2
# Extract the relevant data
poses_3d = data["poses_3d"][track_id]
joint_names = data["joint_names"]

# %%
# Create a list to store the data
data_list = [
    {
        "frame": frame,
        "joint": joint_name,
        "x": pose[0][joint_idx][0],
        "y": pose[0][joint_idx][1],
        "z": pose[0][joint_idx][2],
    }
    for frame, pose in poses_3d.items()
    for joint_idx, joint_name in enumerate(joint_names)
]

# Convert the list to a DataFrame
df = pd.DataFrame(data_list)
# Sort the DataFrame by frame and joint
df = df.sort_values(["frame", "joint"]).reset_index(drop=True)
df.query("joint == 'lank'")["x"].plot()


# %%
def apply_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data)


def apply_gaussian_filter(data, sigma):
    return gaussian_filter1d(data, sigma)


def apply_cubic_spline(x, y, num_points=None):
    if num_points is None:
        num_points = len(x)
    cs = CubicSpline(x, y)
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = cs(x_new)
    return x_new, y_new


# %%
# Parameters for the filters
cutoff = 8.0  # cutoff frequency of the filter, in Hz
fs = 30.0  # sampling rate, in Hz (adjust this to your actual frame rate)
sigma = 2  # standard deviation for Gaussian filter

# Group the data by joint
grouped = df.groupby("joint")

# Create new DataFrames for the smoothed data
df_lowpass = df.copy()
df_gaussian = df.copy()
df_spline_list = []  # We'll use a list to store DataFrames for each joint

# Apply filters to each joint's coordinates
for joint, group in grouped:
    # Sort by frame to ensure time series is in order
    group = group.sort_values("frame")

    # Apply low-pass filter
    df_lowpass.loc[group.index, "x"] = apply_lowpass_filter(
        group["x"].astype(np.float32), cutoff, fs
    )
    df_lowpass.loc[group.index, "y"] = apply_lowpass_filter(
        group["y"].astype(np.float32), cutoff, fs
    )
    df_lowpass.loc[group.index, "z"] = apply_lowpass_filter(
        group["z"].astype(np.float32), cutoff, fs
    )

    # Apply Gaussian filter
    df_gaussian.loc[group.index, "x"] = apply_gaussian_filter(
        group["x"].astype(np.float32), sigma
    )
    df_gaussian.loc[group.index, "y"] = apply_gaussian_filter(
        group["y"].astype(np.float32), sigma
    )
    df_gaussian.loc[group.index, "z"] = apply_gaussian_filter(
        group["z"].astype(np.float32), sigma
    )

    # Apply cubic spline
    frames = group["frame"].values
    # apply it to the low-pass filtered data
    x_new, x_spline = apply_cubic_spline(
        frames, df_lowpass.loc[group.index, "x"]
    )  # group["x"])
    _, y_spline = apply_cubic_spline(frames, group["y"])
    _, z_spline = apply_cubic_spline(frames, group["z"])

    df_spline_list.append(
        pd.DataFrame(
            {
                "frame": x_new,
                "joint": joint,
                "x": x_spline,
                "y": y_spline,
                "z": z_spline,
            }
        )
    )

# Concatenate all the spline DataFrames
df_spline = pd.concat(df_spline_list, ignore_index=True)
df_spline = df_spline.sort_values(["frame", "joint"]).reset_index(drop=True)

# Plot the results for a specific joint (e.g., 'lank')
joint = "lelb"
joint_data = df[df["joint"] == joint].sort_values("frame")
joint_data_lowpass = df_lowpass[df_lowpass["joint"] == joint].sort_values("frame")
joint_data_gaussian = df_gaussian[df_gaussian["joint"] == joint].sort_values("frame")
joint_data_spline = df_spline[df_spline["joint"] == joint].sort_values("frame")

plt.figure(figsize=(15, 5))
plt.plot(joint_data["frame"], joint_data["x"], label="Original")
plt.plot(
    joint_data_lowpass["frame"], joint_data_lowpass["x"], label="Low-pass filtered"
)
plt.plot(
    joint_data_gaussian["frame"], joint_data_gaussian["x"], label="Gaussian filtered"
)
plt.plot(joint_data_spline["frame"], joint_data_spline["x"], label="Cubic Spline")
plt.title(f"X-coordinate of {joint} - Original vs Smoothed")
plt.xlabel("Frame")
plt.ylabel("X-coordinate")
plt.legend()
plt.show()

# %%

# Create a new dictionary to store the smoothed data
smoothed_data = data.copy()

# Apply the low-pass filter to each track
for track_id, track_data in data["poses_3d"].items():
    if not track_data:  # Skip empty tracks
        continue

    smoothed_track = {}
    for frame, pose in track_data.items():
        # Initialize an array to store the smoothed pose
        smoothed_pose = np.zeros_like(pose)

        # Apply the filter to each dimension (x, y, z) separately
        for dim in range(3):
            coords = pose[0, :, dim]
            smoothed_coords = apply_lowpass_filter(coords, cutoff, fs)
            smoothed_pose[0, :, dim] = smoothed_coords

        smoothed_track[frame] = smoothed_pose

    smoothed_data["poses_3d"][track_id] = smoothed_track

# Save the smoothed data to a new pickle file
output_path = "lessort-dunk-01_data_smoothed.pkl"
with open(output_path, "wb") as f:
    pickle.dump(smoothed_data, f)

print(f"Smoothed data saved to {output_path}")
# %%
