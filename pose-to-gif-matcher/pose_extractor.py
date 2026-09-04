import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageSequence

from normalizer import normalize_pose
from pose_matcher import load_database, find_best_match

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(PROJECT_DIR, "database", "poses.json")
GIF_FOLDER = os.path.join(os.path.dirname(__file__), "dataset", "gif")

# Cache so we don't re-read a GIF file from disk on every match
_gif_frame_cache = {}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def draw_pose_landmarks(frame, landmarks, connections):
    height, width, _ = frame.shape

    for connection in connections:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        start_pt = (int(start.x * width), int(start.y * height))
        end_pt = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def get_gif_frames(gif_filename):
    """
    Load and cache all frames of a GIF as BGR numpy arrays
    (OpenCV's native color format), so repeated matches to the
    same GIF don't re-read the file from disk.
    """
    if gif_filename in _gif_frame_cache:
        return _gif_frame_cache[gif_filename]

    gif_path = os.path.join(GIF_FOLDER, gif_filename)
    frames = []

    try:
        img = Image.open(gif_path)
        for frame in ImageSequence.Iterator(img):
            rgb = np.array(frame.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frames.append(bgr)
    except Exception as e:
        print(f"Could not load GIF frames for {gif_filename}: {e}")
        frames = []

    _gif_frame_cache[gif_filename] = frames
    return frames


def combine_with_matches(cam_frame, matches, gif_anim_indices):
    """
    Stack the camera feed next to up to 3 matched GIFs,
    arranged in a vertical column, each labeled with its
    filename and similarity score.
    """
    cam_height = cam_frame.shape[0]
    col_width = 280
    slot_height = cam_height // 3

    gif_column_slots = []

    for entry, score in matches[:3]:
        filename = entry["gif_filename"]
        gif_frames = get_gif_frames(filename)

        if not gif_frames:
            slot = np.zeros((slot_height, col_width, 3), dtype=np.uint8)
        else:
            idx = gif_anim_indices.get(filename, 0)
            idx = idx % len(gif_frames)
            gif_anim_indices[filename] = idx + 1

            resized = cv2.resize(gif_frames[idx], (col_width, slot_height))

            label = f"{filename} ({score:.2f})"
            cv2.putText(
                resized, label, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1
            )
            slot = resized

        gif_column_slots.append(slot)

    # Pad with empty slots if fewer than 3 matches exist
    while len(gif_column_slots) < 3:
        gif_column_slots.append(np.zeros((slot_height, col_width, 3), dtype=np.uint8))

    gif_column = np.vstack(gif_column_slots)

    # In case integer division left a small height mismatch, force-match
    if gif_column.shape[0] != cam_height:
        gif_column = cv2.resize(gif_column, (col_width, cam_height))

    return np.hstack([cam_frame, gif_column])


def main():
    model_path = ensure_model()

    print(f"Loading pose database from {DATABASE_PATH}...")
    database = load_database(DATABASE_PATH)
    print(f"Loaded {len(database)} GIFs.")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Pose Extractor", cv2.WINDOW_NORMAL)

        frame_count = 0
        current_matches = []
        gif_anim_indices = {}

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Failed to grab frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect(mp_image)

                if result.pose_landmarks:
                    pose_landmarks = result.pose_landmarks[0]
                    draw_pose_landmarks(
                        frame,
                        pose_landmarks,
                        mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    )
                    current_pose = normalize_pose(pose_landmarks)

                    frame_count += 1
                    if frame_count % 5 == 0:
                        current_matches = find_best_match(current_pose, database, top_n=3)
                        for entry, score in current_matches:
                            print(f"{entry['gif_filename']}: {score:.2f}")

                if current_matches:
                    display_frame = combine_with_matches(frame, current_matches, gif_anim_indices)
                else:
                    display_frame = frame

                cv2.imshow("Pose Extractor", display_frame)

                if cv2.getWindowProperty("Pose Extractor", cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()