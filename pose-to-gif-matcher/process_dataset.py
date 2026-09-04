import os
import json
import csv

import cv2
import mediapipe as mp

from normalizer import normalize_pose


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VIDEO_FOLDER = os.path.join(DATASET_DIR, "test_videos")
METADATA_FILE = os.path.join(DATASET_DIR, "metadata.txt")

PROJECT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_FILE = os.path.join(PROJECT_DIR, "database", "poses.json")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pose_landmarker_lite.task"
)


# Process approximately one frame every N frames.
#
# Example:
#   30 FPS video
#   FRAME_STEP = 5
#
# means approximately 6 frames per second.
FRAME_STEP = 5


# ==========================================
# MEDIAPIPE
# ==========================================

def create_landmarker():
    """
    Create the MediaPipe Pose Landmarker.

    IMAGE mode is used because we are processing
    previously recorded video frames rather than
    a live webcam stream.
    """

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )


# ==========================================
# READ METADATA
# ==========================================

def load_metadata():
    """
    Load Video2GIF metadata.

    Returns:
        Dictionary mapping YouTube IDs to their
        corresponding GIF metadata.
    """

    metadata = {}

    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found:")
        print(METADATA_FILE)
        return metadata

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        for row in reader:

            youtube_id = row["youtube_id"]

            metadata[youtube_id] = row

    return metadata


# ==========================================
# PROCESS ONE VIDEO
# ==========================================

def process_video(
    landmarker,
    video_path,
    start_sec,
    end_sec
):
    """
    Extract normalized pose vectors from the
    GIF segment of a source video.

    Only frames between start_sec and end_sec
    are processed.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Could not determine video FPS.")
        cap.release()
        return []

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    pose_vectors = []

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Stop once we reach the GIF's end
        if frame_number > end_frame:
            break

        # Ignore frames before GIF segment
        if frame_number < start_frame:
            frame_number += 1
            continue

        # Sample only selected frames
        if (frame_number - start_frame) % FRAME_STEP != 0:
            frame_number += 1
            continue

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Run MediaPipe
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            # Convert landmarks into our
            # normalized 99-value vector
            pose_vector = normalize_pose(
                landmarks
            )

            pose_vectors.append(
                pose_vector
            )

        frame_number += 1

    cap.release()

    return pose_vectors


# ==========================================
# PROCESS DATASET
# ==========================================

def process_dataset():
    """
    Process the Video2GIF test dataset.

    For every available source video:

        video
          ↓
        GIF temporal segment
          ↓
        sampled frames
          ↓
        MediaPipe Pose
          ↓
        normalized pose vectors
          ↓
        poses.json
    """

    print("======================================")
    print("VIDEO2GIF POSE DATASET PROCESSOR")
    print("======================================")
    print()

    # Check dataset folders
    if not os.path.exists(VIDEO_FOLDER):

        print("Video folder does not exist:")
        print(VIDEO_FOLDER)

        return

    # Load metadata
    metadata = load_metadata()

    if not metadata:

        print("No metadata was loaded.")
        return

    print(
        f"Loaded metadata for "
        f"{len(metadata)} videos."
    )

    print()

    dataset = []

    # Create MediaPipe
    with create_landmarker() as landmarker:

        # Process every video in folder
        for filename in sorted(
            os.listdir(VIDEO_FOLDER)
        ):

            if not filename.lower().endswith(
                ".mp4"
            ):
                continue

            video_path = os.path.join(
                VIDEO_FOLDER,
                filename
            )

            # Remove .mp4
            youtube_id = os.path.splitext(
                filename
            )[0]

            print("--------------------------------------")
            print(f"Video: {filename}")
            print(f"YouTube ID: {youtube_id}")

            # Find metadata
            if youtube_id not in metadata:

                print(
                    "No metadata found. Skipping."
                )

                continue

            info = metadata[youtube_id]

            # Get GIF timing
            try:

                start_sec = float(
                    info["gif_start_sec"]
                )

                end_sec = float(
                    info["gif_end_sec"]
                )

            except (
                ValueError,
                KeyError
            ):

                print(
                    "Invalid GIF timing. Skipping."
                )

                continue

            print(
                f"GIF segment: "
                f"{start_sec:.2f}s → "
                f"{end_sec:.2f}s"
            )

            # Process GIF segment
            pose_vectors = process_video(
                landmarker,
                video_path,
                start_sec,
                end_sec
            )

            if not pose_vectors:

                print(
                    "No poses detected."
                )

                continue

            print(
                f"Extracted "
                f"{len(pose_vectors)} poses."
            )

            # Store dataset entry
            entry = {
                "gif_id": info["gif_id"],
                "youtube_id": youtube_id,
                "gif_title": info["gif_title"],
                "gif_start_sec": start_sec,
                "gif_end_sec": end_sec,
                "gif_url": info["gif_url"],
                "poses": pose_vectors
            }

            dataset.append(entry)

    # ======================================
    # SAVE JSON
    # ======================================

    print()
    print("Saving poses.json...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            indent=2
        )

    print()
    print("======================================")
    print("PROCESSING COMPLETE")
    print("======================================")
    print(
        f"Videos processed: {len(dataset)}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )
if __name__ == "__main__":
    process_dataset()