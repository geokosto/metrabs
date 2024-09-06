# %%
# metrabs_to_opencap.py
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
import requests
import json

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


case_name = "pao_promo"
video_name = "mitoglou-move-01"
track_id = 1
datapath = f"clips/{case_name}/data/{video_name}_data.pkl"

df = load_metrabs_data(datapath, case_name, video_name, track_id)

# %% Apply the low-pass filter to the metrabs data


def apply_filter_to_markers(df, cutoff, fs):
    df = df.copy()

    def apply_lowpass_filter(data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        return signal.filtfilt(b, a, data)

    # Get all column names except 'session_id', 'trial_name', and 'Frame'
    marker_columns = [
        col for col in df.columns if col not in ["session_id", "trial_name", "Frame"]
    ]

    # Apply the filter to each marker column
    for col in marker_columns:
        df[col] = apply_lowpass_filter(df[col].values, cutoff, fs)

    return df


# Parameters for the filters
cutoff = 5.0  # cutoff frequency of the filter, in Hz
fs = 30.0  # sampling rate, in Hz (adjust this to your actual frame rate)

# Apply the filter to all marker columns
df_filtered = apply_filter_to_markers(df, cutoff, fs)

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

df = df_filtered

# %% Create a midHip marker from the lhip and rhip markers

df["midHip_X"] = (df["lhip_X"] + df["rhip_X"]) / 2
df["midHip_Y"] = (df["lhip_Y"] + df["rhip_Y"]) / 2
df["midHip_Z"] = (df["lhip_Z"] + df["rhip_Z"]) / 2

# %% Change the cs from inertial world to initial body inertial
# The same code as opencap_data/ml_opensim_markers.py


def transform_coordinates_to_body_cs(
    df,
    origin_marker="pelv",
    left_marker="lhip",
    right_marker="rhip",
    up_marker="neck",
):
    """
    Transform coordinates of markers to a body-centered coordinate system based on the given markers.

    Parameters:
    df (pd.DataFrame): DataFrame containing marker coordinates (X, Y, Z).
    origin_marker (str): The marker to use as the origin of the body coordinate system (default is 'midHip').
    right_marker (str): The marker to use as the right-side reference (default is 'RHip').
    left_marker (str): Optional marker to use as the left-side reference (if provided).
    up_marker (str): Optional marker to use for the vertical axis reference (if provided).

    Returns:
    pd.DataFrame: Transformed coordinates in body-centered coordinate system.
    pd.DataFrame: Euler angles representing the rotation in the body-centered system.
    """

    def calculate_rotation_matrix_old(origin, right, up=None):
        # Z-axis (vertical): from origin to up marker if up_marker is provided
        z_axis = np.array([0, 0, 1])  # Default Z-axis (if no up_marker)
        if up is not None:
            z_axis = up - origin
            z_axis /= np.linalg.norm(z_axis)

        # Y-axis (lateral): from origin to right shoulder
        y_axis = right - origin
        y_axis /= np.linalg.norm(y_axis)

        # X-axis (anterior-posterior): cross product of Y and Z
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return rotation_matrix

    def calculate_rotation_matrix(origin, left, right, up=None):
        # Z-axis (vertical): from origin to up marker if up_marker is provided
        z_axis = np.array([0, 0, 1])  # Default Z-axis (if no up_marker)
        if up is not None:
            z_axis = up - origin
            z_axis /= np.linalg.norm(z_axis)

        # Y-axis (lateral): from origin to left side (left_marker instead of right_marker)
        y_axis = left - origin
        y_axis /= np.linalg.norm(y_axis)

        # X-axis (anterior-posterior): cross product of Y and Z
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return rotation_matrix

    df = df.copy()
    transformed_data = []
    euler_angles = []

    # Get the list of markers present in the DataFrame
    markers = set(
        col.rsplit("_", 1)[0] for col in df.columns if col.endswith(("_X", "_Y", "_Z"))
    )

    # Get initial positions for the markers from the first row
    initial_origin = (
        df[[f"{origin_marker}_X", f"{origin_marker}_Y", f"{origin_marker}_Z"]]
        .iloc[0]
        .values
    )
    initial_left = (
        df[[f"{left_marker}_X", f"{left_marker}_Y", f"{left_marker}_Z"]].iloc[0].values
    )
    initial_right = (
        df[[f"{right_marker}_X", f"{right_marker}_Y", f"{right_marker}_Z"]]
        .iloc[0]
        .values
    )
    initial_up = (
        df[[f"{up_marker}_X", f"{up_marker}_Y", f"{up_marker}_Z"]].iloc[0].values
        if up_marker
        else None
    )

    # Calculate initial rotation matrix
    initial_rotation = calculate_rotation_matrix(
        initial_origin, initial_left, initial_right, initial_up
    )

    for i in range(len(df)):
        # Transform all marker coordinates
        transformed_frame = {}
        for marker in markers:
            marker_coords = (
                df[[f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].iloc[i].values
            )

            # Translate marker coordinates relative to initial origin
            marker_coords -= initial_origin

            # Rotate to initial body-centered CS
            marker_coords = initial_rotation.T @ marker_coords

            # Store transformed coordinates
            transformed_frame[f"{marker}_X"] = marker_coords[0]  # Anterior-Posterior
            transformed_frame[f"{marker}_Y"] = marker_coords[1]  # Lateral
            transformed_frame[f"{marker}_Z"] = marker_coords[2]  # Vertical

        transformed_data.append(transformed_frame)

        # Calculate Euler angles (relative rotation from initial)
        current_origin = (
            df[[f"{origin_marker}_X", f"{origin_marker}_Y", f"{origin_marker}_Z"]]
            .iloc[i]
            .values
        )
        current_left = (
            df[[f"{left_marker}_X", f"{left_marker}_Y", f"{left_marker}_Z"]]
            .iloc[i]
            .values
        )
        current_right = (
            df[[f"{right_marker}_X", f"{right_marker}_Y", f"{right_marker}_Z"]]
            .iloc[i]
            .values
        )
        current_up = (
            df[[f"{up_marker}_X", f"{up_marker}_Y", f"{up_marker}_Z"]].iloc[i].values
            if up_marker
            else None
        )

        current_rotation = calculate_rotation_matrix(
            current_origin, current_left, current_right, current_up
        )
        relative_rotation = current_rotation.T @ initial_rotation

        # Convert to Euler angles
        euler_angles_frame = R.from_matrix(relative_rotation).as_euler(
            "xyz", degrees=True
        )
        euler_angles.append(euler_angles_frame)

    # Create DataFrames for transformed data and Euler angles
    transformed_df = pd.DataFrame(transformed_data, index=df.index)
    euler_df = pd.DataFrame(
        euler_angles, columns=["Euler_X", "Euler_Y", "Euler_Z"], index=df.index
    )

    return transformed_df, euler_df


# Initialize empty lists to store the transformed data and Euler angles
transformed_data_list = []
# euler_data_list = []

# Loop through the grouped data (grouped by session_id and trial_name)
for group in df.groupby(["session_id", "trial_name"]):
    session_id, trial_name = group[0]
    trial_df = group[1]

    # Apply the body-centered transformation
    transformed_df, euler_df = transform_coordinates_to_body_cs(
        trial_df, origin_marker="midHip"
    )

    # Add session and trial columns back to the transformed data
    transformed_df["session_id"] = session_id
    transformed_df["trial_name"] = trial_name
    transformed_df["Frame"] = trial_df["Frame"].values
    # euler_df["session_id"] = session_id
    # euler_df["trial_name"] = trial_name

    # Append the transformed data and Euler angles to the lists
    transformed_data_list.append(transformed_df)
    # euler_data_list.append(euler_df)

# Concatenate the lists of DataFrames into single DataFrames
df_body = pd.concat(transformed_data_list).reset_index(drop=True)
# save the transformed data to pkl file
df_body.to_pickle("data/df_body.pkl")

# %% Load the transformed data from the pkl file
df_body = pd.read_pickle("data/df_body.pkl")

# %% Change units of all the markers from mm to m

# Define the columns to convert
columns_to_convert = [
    col for col in df_body.columns if col not in ["Frame", "session_id", "trial_name"]
]

# Convert the units from mm to m
df_body[columns_to_convert] = df_body[columns_to_convert] * 1e-3

# %% Plot the transformed data for a specific marker using plotly


def plot_marker(df, marker_name):
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

    # Update layout
    fig.update_layout(
        # height=900,
        # width=1000,
        title_text=f"Data for {marker_name}",
    )
    fig.update_xaxes(title_text="Frame", row=3, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=3, col=1)

    # Show the plot
    fig.show()


def plot_hist_per_feature(df):
    features = df.columns
    for feature in features:
        if feature not in ["Frame", "session_id", "trial_name"]:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[feature], name=feature))
            fig.update_layout(title=f"Histogram for {feature}")
            fig.show()


# Example usage:
marker_name = "pelv"  # Replace with the marker you want to plot
# plot_marker(df_body, marker_name)
# plot_hist_per_feature(df_body)


# %% Inference the MetrabsToOpenCap ml model


def mlflow_inference(data, url="http://localhost:5148/invocations"):
    """
    Send a prediction request to a deployed MLflow model endpoint.
    :param data: A pandas DataFrame containing the input data
    :param url: The URL of your deployed model endpoint
    :return: A pandas DataFrame containing the model's predictions
    """
    # Convert DataFrame to list of dictionaries
    instances = data.to_dict(orient="records")

    # Create the JSON payload
    payload = json.dumps({"dataframe_records": instances})

    # Set the content type
    headers = {"Content-Type": "application/json"}

    # Send the request
    response = requests.post(url, data=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()

        # Convert the result to a DataFrame
        if isinstance(result, list):
            predictions = pd.DataFrame(result)
        elif isinstance(result, dict) and "predictions" in result:
            predictions = pd.DataFrame(result["predictions"])
        else:
            raise ValueError(f"Unexpected response format: {result}")

        return predictions
    else:
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.text}"
        )


def rename_reduce_metrabs_to_opencap_keypoints(df):
    # Define the mapping from Metrabs to OpenCap keypoints
    mapping = {
        "neck_X": "Neck_X",
        "neck_Y": "Neck_Y",
        "neck_Z": "Neck_Z",
        "rsho_X": "RShoulder_X",
        "rsho_Y": "RShoulder_Y",
        "rsho_Z": "RShoulder_Z",
        "relb_X": "RElbow_X",
        "relb_Y": "RElbow_Y",
        "relb_Z": "RElbow_Z",
        "rwri_X": "RWrist_X",
        "rwri_Y": "RWrist_Y",
        "rwri_Z": "RWrist_Z",
        "lsho_X": "LShoulder_X",
        "lsho_Y": "LShoulder_Y",
        "lsho_Z": "LShoulder_Z",
        "lelb_X": "LElbow_X",
        "lelb_Y": "LElbow_Y",
        "lelb_Z": "LElbow_Z",
        "lwri_X": "LWrist_X",
        "lwri_Y": "LWrist_Y",
        "lwri_Z": "LWrist_Z",
        # "pelv_X": "midHip_X",
        # "pelv_Y": "midHip_Y",
        # "pelv_Z": "midHip_Z",
        "rhip_X": "RHip_X",
        "rhip_Y": "RHip_Y",
        "rhip_Z": "RHip_Z",
        "rkne_X": "RKnee_X",
        "rkne_Y": "RKnee_Y",
        "rkne_Z": "RKnee_Z",
        "rank_X": "RAnkle_X",
        "rank_Y": "RAnkle_Y",
        "rank_Z": "RAnkle_Z",
        "lhip_X": "LHip_X",
        "lhip_Y": "LHip_Y",
        "lhip_Z": "LHip_Z",
        "lkne_X": "LKnee_X",
        "lkne_Y": "LKnee_Y",
        "lkne_Z": "LKnee_Z",
        "lank_X": "LAnkle_X",
        "lank_Y": "LAnkle_Y",
        "lank_Z": "LAnkle_Z",
    }
    # Define the columns that should be present after renaming
    opencap_metrabs_matched_markers = [
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "RHip",
        "RKnee",
        "RAnkle",
        "LHip",
        "LKnee",
        "LAnkle",
        "midHip",  # i constuct this in metrabs
    ]
    # Create the list of columns you want to keep (i.e., markers + axes)
    selected_columns = [
        f"{col}_{axis}"
        for col in opencap_metrabs_matched_markers
        for axis in ["X", "Y", "Z"]
    ]

    # Rename and reduce the DataFrame to the selected columns
    renamed_df = df.rename(columns=mapping)

    # Filter the DataFrame to keep only the selected columns
    return renamed_df[selected_columns]


# Prepare the input data for the model
input_data = df_body.drop(columns=["session_id", "trial_name", "Frame"])
input_data = rename_reduce_metrabs_to_opencap_keypoints(input_data)

# Call the inference function
output_data = mlflow_inference(input_data)

# Merge the output data with the input data
# df_final = pd.concat(
#     [df_body[["session_id", "trial_name", "Frame"]], output_data], axis=1
# )

df_final = pd.concat(
    [
        df_body[["session_id", "trial_name", "Frame"]],
        input_data,
        output_data,
    ],
    axis=1,
)

# %% Write the output data to a .trc file
# The trc_reader.py is the same as opencap_data/ml_opensim_markers.py

from trc_reader import TRCFile


def df_to_trc(df, output_file, data_rate=30, camera_rate=30, units="mm"):
    # Create a new TRCFile object
    trc_file = TRCFile()

    # Set metadata
    trc_file.path = output_file
    trc_file.data_rate = data_rate
    trc_file.camera_rate = camera_rate
    trc_file.num_frames = len(df)
    trc_file.units = units
    trc_file.orig_data_rate = data_rate
    trc_file.orig_data_start_frame = 1
    trc_file.orig_num_frames = len(df)

    # Extract marker names
    marker_columns = [col for col in df.columns if col.endswith(("_X", "_Y", "_Z"))]
    trc_file.marker_names = list(set([col.rsplit("_", 1)[0] for col in marker_columns]))
    trc_file.num_markers = len(trc_file.marker_names)

    # Prepare data
    trc_file.time = df["Frame"].values.astype(float) / data_rate

    # Create a structured array for the data
    dtype = [("frame_num", "int"), ("time", "float64")]
    for marker in trc_file.marker_names:
        dtype.extend(
            [
                (f"{marker}_tx", "float64"),
                (f"{marker}_ty", "float64"),
                (f"{marker}_tz", "float64"),
            ]
        )

    trc_file.data = np.zeros(trc_file.num_frames, dtype=dtype)
    trc_file.data["frame_num"] = df["Frame"].values
    trc_file.data["time"] = trc_file.time

    for marker in trc_file.marker_names:
        trc_file.data[f"{marker}_tx"] = df[f"{marker}_X"].values
        trc_file.data[f"{marker}_ty"] = df[f"{marker}_Y"].values
        trc_file.data[f"{marker}_tz"] = df[f"{marker}_Z"].values

    # Write to file
    trc_file.write(output_file)


# Write the output data to a .trc file
output_file = f"clips/{case_name}/data/{video_name}_output.trc"

df_to_trc(df_final, output_file, data_rate=30, camera_rate=30, units="m")


# only for testing
df_to_trc(df_body, "test_wo_model.trc", data_rate=30, camera_rate=30, units="m")
df_to_trc(df_final, "test_w_model.trc", data_rate=30, camera_rate=30, units="m")

# %%
raise ValueError("Stop here")

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
