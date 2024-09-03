import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLOv8 model
model_size = "x"  # You can choose different model sizes: n, s, m, l, x
model = YOLO(f"yolov8{model_size}.pt")


def detect_and_track(video_path, output_path, video_output_path):
    cap = cv2.VideoCapture(video_path)
    tracked_boxes = {}

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track humans using YOLOv8
        results = model.track(
            frame, classes=[0], persist=True
        )  # 0 is the class index for persons in COCO dataset

        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()  # x, y, width, height
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                if track_id not in tracked_boxes:
                    tracked_boxes[track_id] = []
                tracked_boxes[track_id].append((frame_count, box))

                # Draw bounding box and track ID
                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()

    # Save tracked boxes
    with open(output_path, "wb") as f:
        pickle.dump(tracked_boxes, f)

    return tracked_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and track humans in a video using YOLOv8."
    )
    parser.add_argument(
        "case_name", type=str, help="Name of the case (e.g., 'pao_promo')."
    )
    parser.add_argument(
        "video_name", type=str, help="Name of the input video file (without extension)."
    )
    args = parser.parse_args()

    # Define paths based on the new directory structure
    video_name = args.video_name
    case_name = args.case_name
    video_path = os.path.join("clips", case_name, "raw", f"{video_name}")
    video_name = video_name.split(".")[0]
    output_path = os.path.join(
        "clips", case_name, "data", f"{video_name}_tracked_boxes.pkl"
    )
    video_output_path = os.path.join(
        "clips", case_name, "yolo_boxes", f"{video_name}_output_boxes.mp4"
    )

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

    tracked_boxes = detect_and_track(video_path, output_path, video_output_path)

    print(f"Tracking completed. Results saved to {output_path}")
    print(f"Output video saved to {video_output_path}")
    print(f"Number of tracked individuals: {len(tracked_boxes)}")
    for track_id, boxes in tracked_boxes.items():
        print(f"Track ID {track_id}: {len(boxes)} detections")
