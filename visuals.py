# %%

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML

import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation as R

matplotlib.rcParams["animation.embed_limit"] = 500  # Increase limit to 50 MB


def plot_2d_markers(
    X,
    xaxis,
    yaxis,
    merged_data,
    session_trial,
    marker="midHip",
    title="Marker Positions",
    show=True,
):
    # Filter the data by the specified session_trial
    session_trial_data = merged_data[merged_data["session_trial"] == session_trial]
    indices = session_trial_data.index
    X_filtered = X.loc[indices]

    fig = go.Figure()

    # Plot the specified marker with color representing time
    fig.add_trace(
        go.Scatter(
            x=X_filtered[f"{marker}_{xaxis}"],
            y=X_filtered[f"{marker}_{yaxis}"],
            mode="lines+markers",
            marker=dict(
                color=session_trial_data["Time"], colorscale="Viridis", showscale=True
            ),
            name=marker,
        )
    )

    fig.update_layout(
        title=f"{title} - {session_trial} - {marker}",
        xaxis_title=f"{xaxis} Position",
        yaxis_title=f"{yaxis} Position",
        legend_title="Markers",
        width=800,
        height=600,
    )

    if show:
        fig.show()


def plot_3d_scatter_marker(
    df, session_id, trial_name, marker, title="3D Scatter Plot", save=False
):
    # Filter the data by the specified session_trial
    session_trial_data = df[
        (df["session_id"] == session_id) & (df["trial_name"] == trial_name)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=session_trial_data[f"{marker}_X"],
            y=session_trial_data[f"{marker}_Y"],
            z=session_trial_data[f"{marker}_Z"],
            mode="lines+markers",
            marker=dict(
                color=session_trial_data["Frame"], colorscale="Viridis", showscale=True
            ),
            name=marker,
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        # width=800,
        # height=600,
    )

    if save:
        fig.write_html(f"3d_scatter_{session_id}-{trial_name}-{title}.html")
    else:
        fig.show()


# %%


def create_3d_animation(
    X_transformed, euler_angles, merged_data, session_trial, marker="midHip"
):
    # Filter the data by the specified session_trial
    session_trial_data = merged_data[merged_data["session_trial"] == session_trial]
    indices = session_trial_data.index
    X_filtered = X_transformed.loc[indices]
    euler_filtered = euler_angles.loc[indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(num, X_filtered, euler_filtered, marker):
        ax.clear()
        ax.scatter(
            X_filtered[f"{marker}_X"].iloc[:num],
            X_filtered[f"{marker}_Y"].iloc[:num],
            X_filtered[f"{marker}_Z"].iloc[:num],
            c=range(num),
            cmap="viridis",
        )
        current_x = X_filtered[f"{marker}_X"].iloc[num]
        current_y = X_filtered[f"{marker}_Y"].iloc[num]
        current_z = X_filtered[f"{marker}_Z"].iloc[num]

        ax.quiver(
            current_x,
            current_y,
            current_z,
            np.cos(np.radians(euler_filtered["Euler_X"].iloc[num])),
            np.cos(np.radians(euler_filtered["Euler_Y"].iloc[num])),
            np.cos(np.radians(euler_filtered["Euler_Z"].iloc[num])),
            length=1.0,
            normalize=True,
        )

        ax.set_xlim([X_filtered[f"{marker}_X"].min(), X_filtered[f"{marker}_X"].max()])
        ax.set_ylim([X_filtered[f"{marker}_Y"].min(), X_filtered[f"{marker}_Y"].max()])
        ax.set_zlim([X_filtered[f"{marker}_Z"].min(), X_filtered[f"{marker}_Z"].max()])
        ax.set_title(f"3D Scatter Plot - {marker} - Frame {num}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(X_filtered),
        fargs=(X_filtered, euler_filtered, marker),
        interval=100,
    )
    return ani


# %%


def create_3d_skeleton_animation(
    X_transformed, euler_angles, merged_data, session_trial
):
    # Define marker connections for the skeleton
    connections = [
        ("Head", "Neck"),
        ("Neck", "RightShoulder"),
        ("Neck", "LeftShoulder"),
        ("RightShoulder", "RightElbow"),
        ("RightElbow", "RightWrist"),
        ("LeftShoulder", "LeftElbow"),
        ("LeftElbow", "LeftWrist"),
        ("Neck", "SpineShoulder"),
        ("SpineShoulder", "SpineMid"),
        ("SpineMid", "SpineBase"),
        ("SpineBase", "RightHip"),
        ("SpineBase", "LeftHip"),
        ("RightHip", "RightKnee"),
        ("RightKnee", "RightAnkle"),
        ("LeftHip", "LeftKnee"),
        ("LeftKnee", "LeftAnkle"),
    ]

    # Filter the data by the specified session_trial
    session_trial_data = merged_data[merged_data["session_trial"] == session_trial]
    indices = session_trial_data.index
    X_filtered = X_transformed.loc[indices]
    euler_filtered = euler_angles.loc[indices]

    # Calculate overall min and max for each axis
    x_min, x_max = (
        X_filtered.filter(like="_X").min().min(),
        X_filtered.filter(like="_X").max().max(),
    )
    y_min, y_max = (
        X_filtered.filter(like="_Y").min().min(),
        X_filtered.filter(like="_Y").max().max(),
    )
    z_min, z_max = (
        X_filtered.filter(like="_Z").min().min(),
        X_filtered.filter(like="_Z").max().max(),
    )

    # Add a small margin to the limits
    margin = 0.1
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
    y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
    z_min, z_max = z_min - margin * z_range, z_max + margin * z_range

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(num):
        ax.clear()

        # Plot all markers
        markers = [
            col.split("_")[0] for col in X_filtered.columns if col.endswith("_X")
        ]
        for marker in markers:
            ax.scatter(
                X_filtered[f"{marker}_X"].iloc[num],
                X_filtered[f"{marker}_Y"].iloc[num],
                X_filtered[f"{marker}_Z"].iloc[num],
                c="b",
                s=20,
            )

        # Draw connections
        for start, end in connections:
            if all(f"{marker}_X" in X_filtered.columns for marker in [start, end]):
                ax.plot(
                    [
                        X_filtered[f"{start}_X"].iloc[num],
                        X_filtered[f"{end}_X"].iloc[num],
                    ],
                    [
                        X_filtered[f"{start}_Y"].iloc[num],
                        X_filtered[f"{end}_Y"].iloc[num],
                    ],
                    [
                        X_filtered[f"{start}_Z"].iloc[num],
                        X_filtered[f"{end}_Z"].iloc[num],
                    ],
                    c="r",
                )

        # Plot orientation axes for midHip
        current_x = X_filtered["midHip_X"].iloc[num]
        current_y = X_filtered["midHip_Y"].iloc[num]
        current_z = X_filtered["midHip_Z"].iloc[num]

        # Calculate rotation matrix from Euler angles
        rx, ry, rz = np.radians(euler_filtered.iloc[num])

        R_mat = R.from_euler("xyz", [rx, ry, rz]).as_matrix()

        # Define axis vectors
        axis_length = 0.5
        x_axis = R_mat @ np.array([axis_length, 0, 0])
        y_axis = R_mat @ np.array([0, axis_length, 0])
        z_axis = R_mat @ np.array([0, 0, axis_length])

        # Plot XYZ axes
        ax.quiver(
            current_x,
            current_y,
            current_z,
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            length=axis_length,
            normalize=True,
        )
        ax.quiver(
            current_x,
            current_y,
            current_z,
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            length=axis_length,
            normalize=True,
        )
        ax.quiver(
            current_x,
            current_y,
            current_z,
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            length=axis_length,
            normalize=True,
        )

        # Set fixed axis limits to show entire capture space
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_title(f"3D Skeleton Animation - Frame {num}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")

        # Optional: add a text annotation showing current frame
        ax.text2D(0.05, 0.95, f"Frame: {num}", transform=ax.transAxes)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(X_filtered),
        interval=50,
    )

    return ani
