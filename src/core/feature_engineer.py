"""Do feature engineering like Joint angles, Joint distances, PCA."""
import itertools

import numpy as np

from src.utils.const import ANGLE_PAIRS
from typing import Tuple, List


def keypoint_to_vectors(
        first_point: np.ndarray,
        second_point: np.ndarray,
        third_point: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vector between 2 points with second_point as a reference point.

    Args:
        first_point (np.ndarray): [2, ] First keypoint.
        second_point (np.ndarray): [2, ] Second keypoint.
        third_point (np.ndarray): [2, ] Third keypoint.

    Returns:
        Tuple contains two vector caluculated from 3 points.
    """
    first_vector = second_point - first_point
    second_vector = second_point - third_point

    return (first_vector, second_vector)


def compute_angle(
        first_vector: np.ndarray,
        second_vector: np.ndarray,
    ) -> float:
    """Calculate angle between 2 vectors.

    Args:
        first_vector (np.ndarray): [2, ] First vector.
        second_vector (np.ndarray): [2, ] Second vector.

    Returns:
        An angle computed from 2 vectors.
    """
    unit_first_vector = first_vector / np.linalg.norm(first_vector)
    unit_second_vector = second_vector / np.linalg.norm(second_vector)
    dot_product = unit_first_vector.dot(unit_second_vector.T)
    angle = np.rad2deg(np.arccos(dot_product))

    return angle


def get_angles(list_keypoints: np.ndarray) -> List[float]:
    """Get all angles between define pair of joints.

    Args:
        list_keypoints (np.ndarray): [N, 17, 2] Keypoint array.

    Returns:
        List of float contrain all angles N rows, 24 features.
    """
    angles_list = []
    for keypoints in list_keypoints:
        for angle_pair in ANGLE_PAIRS:
            first_keypoint = keypoints[angle_pair[0]]
            second_keypoint = keypoints[angle_pair[1]]
            third_keypoint = keypoints[angle_pair[2]]

            joint_vector = keypoint_to_vectors(
                            first_point=first_keypoint,
                            second_point=second_keypoint,
                            third_point=third_keypoint
            )

            angle = compute_angle(
                first_vector=joint_vector[0], second_vector=joint_vector[1]
            )

            angles_list.append(angle)

    return np.array(angles_list).reshape(-1, 24)


def compute_distance(keypoints: np.ndarray) -> np.ndarray:
    """Get all euclidience distance between pair of rows.

    Args:
        keypoints (np.ndarray): [17, 2] Keypoint array.

    Returns:
        Numpy array contain distance between each joint with
        16 remaining joints.
        17 * (17-1)/2 = 136 features
    """
    distances = []

    for iters in itertools.combinations(keypoints, 2):
        distances.append(np.linalg.norm(iters[0] - iters[1]))

    return np.array(distances)

def get_distances(list_keypoints: np.ndarray) -> np.ndarray:
    """Get all euclidience distance between pair of rows.

    Args:
        list_keypoints (np.ndarray): [N, 17, 2] Keypoint array.

    Returns:
        Numpy array contain distance between each joint with
        16 remaining joints.
        17 * (17-1)/2 = 136 features
    """
    all_distances = []

    for keypoints in list_keypoints:
        all_distances.append(compute_distance(keypoints))

    return np.array(all_distances)
