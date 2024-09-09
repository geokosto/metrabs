# %%
# smooth_markers.py
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle

# %% Given name of the case, video, and track ID, load the metrabs data
case_name = "pao_promo"
video_name = "hernagomez-move-01"
track_id = 1
datapath = f"clips/{case_name}/data/{video_name}_data.pkl"

# %% Load the metrabs pkl data


def load_metrabs_data(datapath, case_name, video_name, track_id):
    # Load the metrabs pkl data
    with open(datapath, "rb") as f:
        data = pd.read_pickle(f)

    # Extract the relevant data
    poses_3d = data["poses_3d"][track_id]
    joint_names = data["joint_names"]

    # Create DataFrame directly from the data
    df = pd.DataFrame(
        [
            {
                "Frame": frame,
                "joint": joint_name,
                "x": pose[0][joint_idx][0],
                "y": pose[0][joint_idx][1],
                "z": pose[0][joint_idx][2],
                "session_id": case_name,
                "trial_name": video_name,
            }
            for frame, pose in poses_3d.items()
            for joint_idx, joint_name in enumerate(joint_names)
        ]
    )

    # Pivot the data
    df_pivot = df.pivot(
        index=["session_id", "trial_name", "Frame"],
        columns="joint",
        values=["x", "y", "z"],
    )

    # Flatten and rename columns
    df_pivot.columns = [f"{col[1]}_{col[0].upper()}" for col in df_pivot.columns]

    # Reset index and sort
    df_final = df_pivot.reset_index().sort_values("Frame")

    # Reorder columns
    column_order = ["session_id", "trial_name", "Frame"] + [
        col
        for col in df_final.columns
        if col not in ["session_id", "trial_name", "Frame"]
    ]
    df_final = df_final[column_order]

    return df_final.reset_index(drop=True)


# Load the original data
with open(datapath, "rb") as f:
    original_data = pickle.load(f)

df = load_metrabs_data(datapath, case_name, video_name, track_id)
# %% Apply the low-pass filter to the metrabs data


def apply_filter_to_markers(df, cutoff, fs, gradients_threshold=0.1):
    df_filtered = df.copy()

    def handle_outliers_and_interpolate(data):
        # Detect outliers
        mean = data.mean()
        std = data.std()
        is_outlier = (data - mean).abs() > 3 * std

        # Count and print outliers
        num_outliers = is_outlier.sum()
        # if num_outliers > 0:
        # print(f"Column {data.name}: {num_outliers} outliers detected")
        # print(f"  Min: {data.min()}, Max: {data.max()}, Mean: {mean}, Std: {std}")

        # Replace outliers with NaN
        data_cleaned = data.copy()
        data_cleaned[is_outlier] = np.nan

        # print(data_cleaned.head(10))

        # Interpolate
        data_interpolated = data_cleaned.interpolate(method="polynomial", order=2)

        # Handle edge cases
        # data_interpolated = data_interpolated.fillna(method="ffill").fillna(
        #     method="bfill"
        # )
        data_interpolated = data_interpolated.ffill().bfill()

        # print(f"  Interpolated {data_interpolated.isnull().sum()} missing values")
        # print(data_interpolated.head(10))
        return data_interpolated

    def apply_lowpass_filter(data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, data)

    def normalize_data(data):
        min_val = np.min(data)
        max_val = np.max(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1, min_val, max_val

    def denormalize_data(data, min_val, max_val):
        return (data + 1) * (max_val - min_val) / 2 + min_val

    def filter_large_gradients(data, threshold):
        normalized_data, min_val, max_val = normalize_data(data)
        gradients = np.abs(np.diff(normalized_data))
        large_gradients = np.where(gradients > threshold)[0]

        if len(large_gradients) > 0:
            print(f"  {len(large_gradients)} large gradients detected")
            data_filtered = normalized_data.copy()
            data_filtered[large_gradients] = np.nan
            data_filtered[large_gradients + 1] = np.nan

            # Interpolate the removed points
            data_filtered = pd.Series(data_filtered).interpolate(method="cubic")

            # Handle edge cases
            # data_filtered = data_filtered.fillna(method="ffill").fillna(method="bfill")
            data_filtered = data_filtered.ffill().bfill()

            # Denormalize the data
            return denormalize_data(data_filtered.values, min_val, max_val)
        return data

    # Get all column names except 'session_id', 'trial_name', and 'Frame'
    marker_columns = [
        col
        for col in df_filtered.columns
        if col not in ["session_id", "trial_name", "Frame"]
    ]

    # Apply outlier handling, interpolation, and then the filter to each marker column
    for col in marker_columns:
        # Handle outliers and interpolate
        df_filtered[col] = handle_outliers_and_interpolate(df_filtered[col]).values

        # Filter large gradients
        df_filtered[col] = filter_large_gradients(
            df_filtered[col].values, gradients_threshold
        )

        # Apply the lowpass filter
        df_filtered[col] = apply_lowpass_filter(df_filtered[col].values, cutoff, fs)

    return df_filtered


# Parameters for the filters
cutoff = 5.0  # cutoff frequency of the filter, in Hz
fs = 30.0  # sampling rate, in Hz (adjust this to your actual frame rate)
gradients_threshold = 1  # threshold for detecting large gradients
# Apply the filter to all marker columns
df_filtered = apply_filter_to_markers(
    df, cutoff, fs, gradients_threshold=gradients_threshold
)

# %% Plot the original and filtered data for a specific marker using plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_original_vs_filtered(df, df_filtered, marker_name):
    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{marker_name}_X", f"{marker_name}_Y", f"{marker_name}_Z"),
    )

    # Plot data for X, Y, and Z
    for i, axis in enumerate(["X", "Y", "Z"]):
        col_name = f"{marker_name}_{axis}"

        # Original data
        fig.add_trace(
            go.Scatter(
                x=df["Frame"], y=df[col_name], mode="lines", name=f"Original {axis}"
            ),
            row=i + 1,
            col=1,
        )

        # Filtered data
        fig.add_trace(
            go.Scatter(
                x=df_filtered["Frame"],
                y=df_filtered[col_name],
                mode="lines",
                name=f"Filtered {axis}",
            ),
            row=i + 1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        # height=900,
        # width=1000,
        title_text=f"Comparison of Original and Filtered Data for {marker_name}",
    )
    fig.update_xaxes(title_text="Frame", row=3, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)

    # Show the plot
    fig.show()


# Example usage:
marker_name = "pelv"  # Replace with the marker you want to plot
plot_original_vs_filtered(df, df_filtered, marker_name)


# %% # Save the filtered data back to a pickle file in the original MeTRAbs format


def save_metrabs_data(df, datapath, track_id):
    """
    Save a DataFrame back to a pickle file in the original MeTRAbs format,
    preserving all original data structure and joint order.

    Args:
    df (pandas.DataFrame): The DataFrame to save
    datapath (str): The path where the pickle file should be saved
    track_id (int): The track ID to use in the saved data

    Returns:
    None
    """
    # Load the original data to get the correct joint order
    with open(datapath, "rb") as f:
        original_data = pickle.load(f)

    # Get the original joint order
    original_joint_names = original_data["joint_names"]

    # Transform the DataFrame back to the original format
    poses_3d = {}

    for _, row in df.iterrows():
        frame = int(row["Frame"])
        pose = np.zeros((1, len(original_joint_names), 3), dtype=np.float32)
        for i, joint in enumerate(original_joint_names):
            pose[0, i] = [row[f"{joint}_X"], row[f"{joint}_Y"], row[f"{joint}_Z"]]
        poses_3d[frame] = pose

    # Create the data dictionary, preserving the original structure
    data = original_data.copy()

    # Update poses_3d for the specific track_id
    if "poses_3d" not in data:
        data["poses_3d"] = {}
    data["poses_3d"][track_id] = poses_3d

    # Save the data to a pickle file
    datapath_out = datapath.split(".")[0] + "_filtered.pkl"

    with open(datapath_out, "wb") as f:
        pickle.dump(data, f)

    print(f"Data successfully saved to {datapath_out}")

    return data


filtered_data = save_metrabs_data(df_filtered, datapath, track_id)

# %%
raise Exception("Stop here")

# %%
# Set track id
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
