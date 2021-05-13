"""Do feature engineering like Joint angles, Joint distances, PCA."""

import itertools
import numpy as np

from typing import Tuple


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


def get_angle(
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
    dot_product = unit_first_vector.dot(unit_second_vector)
    angle = np.rad2deg(np.arccos(dot_product))

    return angle


def get_distances(keypoints: np.ndarray) -> np.ndarray:
    """Calculate euclidience distance between pair of rows.

    Args:
        keypoints (np.ndarray): [17, 2] Keypoint array.

    Returns:
        A numpy array contain distance between each joint with
        16 remaining joints.
        17 * (17-1)/2 = 136 features
    """
    all_distances = []

    for iters in itertools.combinations(keypoints, 2):
        all_distances.append(np.linalg.norm(iters[0] - iters[1]))

    return np.array(all_distances)
