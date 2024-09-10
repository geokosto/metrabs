# %%
# smooth_markers.py
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle

# %%


def load_metrabs_data(datapath, case_name, video_name):
    # Load the metrabs pkl data
    with open(datapath, "rb") as f:
        data = pd.read_pickle(f)

    # Extract the relevant data
    poses_3d = data["poses_3d"]
    joint_names = data["joint_names"]

    all_tracks_df = []

    for track_id, track_poses in poses_3d.items():
        track_data = []
        for frame, pose_array in track_poses.items():
            if pose_array.size > 0:  # Check if the pose array is not empty
                for joint_idx, joint_name in enumerate(joint_names):
                    track_data.append(
                        {
                            "Frame": frame,
                            "joint": joint_name,
                            "x": pose_array[0, joint_idx, 0],
                            "y": pose_array[0, joint_idx, 1],
                            "z": pose_array[0, joint_idx, 2],
                            "track_id": track_id,
                            "session_id": case_name,
                            "trial_name": video_name,
                        }
                    )

        if track_data:  # Only create a DataFrame if there's data
            df = pd.DataFrame(track_data)

            # Pivot the data
            df_pivot = df.pivot(
                index=["session_id", "trial_name", "track_id", "Frame"],
                columns="joint",
                values=["x", "y", "z"],
            )

            # Flatten and rename columns
            df_pivot.columns = [
                f"{col[1]}_{col[0].upper()}" for col in df_pivot.columns
            ]

            # Reset index and sort
            df_final = df_pivot.reset_index().sort_values(["track_id", "Frame"])

            # Reorder columns
            column_order = ["session_id", "trial_name", "track_id", "Frame"] + [
                col
                for col in df_final.columns
                if col not in ["session_id", "trial_name", "track_id", "Frame"]
            ]
            df_final = df_final[column_order]

            all_tracks_df.append(df_final)

    # Concatenate all track DataFrames
    if all_tracks_df:
        return pd.concat(all_tracks_df, ignore_index=True)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data


def apply_filter_to_markers(
    df, cutoff, fs, gradients_threshold=0.1, outlier_threshold=3
):
    df_filtered = df.copy()

    def handle_outliers_and_interpolate(data, threshold=outlier_threshold):
        mean = data.mean()
        std = data.std()
        is_outlier = (data - mean).abs() > threshold * std
        data_cleaned = data.copy()
        data_cleaned[is_outlier] = np.nan
        data_interpolated = data_cleaned.interpolate(method="polynomial", order=2)
        return data_interpolated.ffill().bfill()

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
            data_filtered = normalized_data.copy()
            data_filtered[large_gradients] = np.nan
            data_filtered[large_gradients + 1] = np.nan
            data_filtered = pd.Series(data_filtered).interpolate(method="cubic")
            data_filtered = data_filtered.ffill().bfill()
            return denormalize_data(data_filtered.values, min_val, max_val)
        return data

    marker_columns = [
        col
        for col in df_filtered.columns
        if col not in ["session_id", "trial_name", "track_id", "Frame"]
    ]

    for track_id in df_filtered["track_id"].unique():
        track_mask = df_filtered["track_id"] == track_id
        for col in marker_columns:
            df_filtered.loc[track_mask, col] = handle_outliers_and_interpolate(
                df_filtered.loc[track_mask, col], outlier_threshold
            ).values
            df_filtered.loc[track_mask, col] = filter_large_gradients(
                df_filtered.loc[track_mask, col].values, gradients_threshold
            )
            df_filtered.loc[track_mask, col] = apply_lowpass_filter(
                df_filtered.loc[track_mask, col].values, cutoff, fs
            )

    return df_filtered


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


def save_metrabs_data(df, datapath):
    with open(datapath, "rb") as f:
        original_data = pickle.load(f)

    original_joint_names = original_data["joint_names"]

    poses_3d = {}

    for track_id in df["track_id"].unique():
        track_df = df[df["track_id"] == track_id]
        track_poses = {}
        for _, row in track_df.iterrows():
            frame = int(row["Frame"])
            pose = np.zeros((1, len(original_joint_names), 3), dtype=np.float32)
            for i, joint in enumerate(original_joint_names):
                pose[0, i] = [row[f"{joint}_X"], row[f"{joint}_Y"], row[f"{joint}_Z"]]
            track_poses[frame] = pose
        poses_3d[track_id] = track_poses

    data = original_data.copy()
    data["poses_3d"] = poses_3d

    datapath_out = datapath.split(".")[0] + "_filtered.pkl"

    with open(datapath_out, "wb") as f:
        pickle.dump(data, f)

    print(f"Data successfully saved to {datapath_out}")

    return data


# %% Usage


case_name = "pao_promo"
video_name = "lessort-dunk-02"
datapath = f"clips/{case_name}/data/{video_name}_data.pkl"

# Load data for all tracks
df = load_metrabs_data(datapath, case_name, video_name)

print(f"{df.trial_name.unique()=}")

# Apply filter to all tracks
cutoff = 5.0
fs = 30.0
gradients_threshold = 0.3
outlier_threshold = 3
df_filtered = apply_filter_to_markers(
    df,
    cutoff,
    fs,
    gradients_threshold=gradients_threshold,
    outlier_threshold=outlier_threshold,
)

# Save filtered data for all tracks
filtered_data = save_metrabs_data(df_filtered, datapath)

# marker_name = "pelv"  # Replace with the marker you want to plot
plot_original_vs_filtered(df, df_filtered, "pelv")


# %%
