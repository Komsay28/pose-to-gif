import math


def normalize_pose(landmarks):
    """
    Convert MediaPipe pose landmarks into a normalized pose vector.

    The pose is normalized relative to the center of the hips
    and scaled according to the distance between the shoulders.
    """

    # MediaPipe landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    # Get important reference points
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]

    left_hip = landmarks[LEFT_HIP]
    right_hip = landmarks[RIGHT_HIP]

    # Calculate the center of the hips
    center_x = (left_hip.x + right_hip.x) / 2
    center_y = (left_hip.y + right_hip.y) / 2
    center_z = (left_hip.z + right_hip.z) / 2

    # Calculate shoulder width
    shoulder_width = math.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2
    )

    # Prevent division by zero
    if shoulder_width == 0:
        shoulder_width = 1

    pose_vector = []

    for landmark in landmarks:
        # Move the pose relative to the hip center
        x = (landmark.x - center_x) / shoulder_width
        y = (landmark.y - center_y) / shoulder_width
        z = (landmark.z - center_z) / shoulder_width

        pose_vector.extend([x, y, z])

    return pose_vector