KEYPOINT_PAIRS = [
    [15, 13], [13, 11], [16, 14], [14, 12],
    [11, 12], [5, 11], [6, 12], [5, 6],
    [5, 7], [6, 8], [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3],
    [2, 4], [3, 5], [4, 6],
]

KEYPOINT_COLS = [
    "nose", "left_eye", "right_eye", "left_ear",
    "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

ANGLE_PAIRS = [
    (5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16),
    (7, 5, 11), (8, 6, 12), (5, 11, 13), (6, 12, 14),
    (1, 3, 5), (2, 4, 6), (3, 5, 6), (4, 6, 5),
    (6, 7, 5), (5, 6, 8), (5, 9, 7), (6, 10, 8),
    (11, 15, 13), (12, 16, 14), (12, 11, 13), (11, 12, 14),
    (11, 13, 12), (11, 14, 12), (13, 11, 15), (14, 12, 16)
]

ANGLE_COLS = [
    "left_elbow", "right_elbow", "left_knee", "right_knee",
    "left_arm", "right_arm", "left_leg", "right_leg",
    "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_shoulder2", "right_shoulder2", "left_hand", "right_hand",
    "left_foot", "right_foot", "left_thigh", "right_thigh",
    "left_knee2", "right_knee2", "left_thigh2", "right_thigh2"
]