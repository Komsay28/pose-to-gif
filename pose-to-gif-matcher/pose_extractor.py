import ctypes
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core import mediapipe_c_bindings
from normalizer import normalize_pose
from pose_matcher import cosine_similarity

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")


def ensure_mediapipe_windows_compat():
    if os.name != "nt":
        return

    try:
        mediapipe_c_bindings._shared_lib.free
    except AttributeError:
        import importlib.resources as resources

        lib_path = resources.files("mediapipe.tasks.c") / "libmediapipe.dll"
        lib = ctypes.CDLL(str(lib_path))
        lib.free = ctypes.CDLL("msvcrt.dll").free
        lib.free.argtypes = [ctypes.c_void_p]
        lib.free.restype = None
        mediapipe_c_bindings._shared_lib = lib


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


def main():
    ensure_mediapipe_windows_compat()
    model_path = ensure_model()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        reference_pose = None
        cv2.namedWindow("Pose Extractor", cv2.WINDOW_NORMAL)

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
                current_pose = None

                if result.pose_landmarks:
                    pose_landmarks = result.pose_landmarks[0]
                    draw_pose_landmarks(
                        frame,
                        pose_landmarks,
                        mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    )
                    current_pose = normalize_pose(pose_landmarks)
                cv2.imshow("Pose Extractor", frame)

                if cv2.getWindowProperty("Pose Extractor", cv2.WND_PROP_VISIBLE) < 1:
                        break

                key = cv2.waitKey(5) & 0xFF

                if key == ord("s") and current_pose is not None:
                    reference_pose = current_pose
                    print("Reference pose saved.")
                    print("Pose vector:")
                    print(reference_pose)
                    print("Number of values:", len(reference_pose))

                if reference_pose is not None and current_pose is not None:
                    similarity = cosine_similarity(current_pose, reference_pose)
                    print(f"Pose similarity: {similarity:.4f}")

                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
