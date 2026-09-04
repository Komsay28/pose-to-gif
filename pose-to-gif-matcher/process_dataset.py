import os
import json

from PIL import Image, ImageSequence
import numpy as np
import mediapipe as mp

from normalizer import normalize_pose

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GIF_FOLDER = os.path.join(BASE_DIR, "dataset", "gif")

PROJECT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_FILE = os.path.join(PROJECT_DIR, "database", "poses.json")

MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

# Process every Nth frame of each GIF (some GIFs have 30-60+ frames)
FRAME_STEP = 2


# ==========================================
# MEDIAPIPE
# ==========================================

def create_landmarker():
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


# ==========================================
# PROCESS ONE GIF
# ==========================================

def process_gif(landmarker, gif_path):
    """
    Extract normalized pose vectors from every Nth frame of a GIF.
    """
    pose_vectors = []

    try:
        img = Image.open(gif_path)
    except Exception as e:
        print(f"Could not open GIF: {gif_path} ({e})")
        return []

    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i % FRAME_STEP != 0:
            continue

        # Convert to RGB numpy array (GIF frames are often palette-mode)
        rgb_frame = np.array(frame.convert("RGB"))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            pose_vector = normalize_pose(landmarks)
            pose_vectors.append(pose_vector)

    return pose_vectors


# ==========================================
# PROCESS DATASET
# ==========================================

def process_dataset():
    print("======================================")
    print("GIF POSE DATASET PROCESSOR")
    print("======================================")
    print()

    if not os.path.exists(GIF_FOLDER):
        print("GIF folder does not exist:")
        print(GIF_FOLDER)
        return

    dataset = []

    with create_landmarker() as landmarker:
        for filename in sorted(os.listdir(GIF_FOLDER)):
            if not filename.lower().endswith(".gif"):
                continue

            gif_path = os.path.join(GIF_FOLDER, filename)
            gif_id = os.path.splitext(filename)[0]

            print("--------------------------------------")
            print(f"GIF: {filename}")

            pose_vectors = process_gif(landmarker, gif_path)

            if not pose_vectors:
                print("No poses detected. Skipping.")
                continue

            print(f"Extracted {len(pose_vectors)} poses.")

            entry = {
                "gif_id": gif_id,
                "gif_filename": filename,
                "poses": pose_vectors,
            }

            dataset.append(entry)

    print()
    print("Saving poses.json...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2)

    print()
    print("======================================")
    print("PROCESSING COMPLETE")
    print("======================================")
    print(f"GIFs processed: {len(dataset)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_dataset()