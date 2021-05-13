""" Generate ground truth .csv from annotation."""
import glob
import os

import pandas as pd
import numpy as np

from src.core.feature_engineer import get_angles
from src.core.feature_engineer import get_distances

from src.utils import get_json
from src.utils.const import KEYPOINT_COLS
from src.utils.const import ANGLE_COLS
from src.utils.const import  JOINT_DISTANCE_COLS
from typing import Dict, List


def make_xy_column(column: List):
    """Add x, y in last column name.

    Args:
        column (list): Column name.
    Returns:
        A list contain column name with x and y at last.
    """
    xy_column = []
    for col_name in column:
        xy_column.append(col_name + '_x')
        xy_column.append(col_name + '_y')

    return xy_column


def create_dataframe(
        keypoints: np.ndarray,
        filenames: List[str],
        classnames: List[str],
        angles: List[float],
        joint_distances: List[float],
    ) -> pd.DataFrame:
    """Create ground truth dataframe.

    Args:
        keypoints (np.ndarray): [N, 17, 2] Keypoint in x, y coordinate.
        filenames (List[str]): List of file name.
        classnames (List[str]): List of class name.
        angles (List[float]): List of joint angles.
        joint_distances (List[float]): List of joint distances.

    Returns:
        Dataframe contain ... features. with label and file names.
    """
    KEYPOINT_COLS_XY = make_xy_column(KEYPOINT_COLS)

    keypoints_flatten = keypoints.reshape(-1, 17 * 2)
    df_keypoint = pd.DataFrame(keypoints_flatten, columns=KEYPOINT_COLS_XY)

    df_angles = pd.DataFrame(angles, columns=ANGLE_COLS)
    df_distances = pd.DataFrame(joint_distances, columns=JOINT_DISTANCE_COLS)

    df = pd.concat([df_keypoint, df_angles, df_distances], axis=1)
    df['filename'] = filenames
    df['class_name'] = classnames

    return df


def main():
    current_dir = os.path.dirname(__file__)
    annotation_dir = '../data/processed/action/annotations'
    save_dir = '../data/processed/action/csv/ground_truth.csv'

    full_annotation_dir = os.path.abspath(os.path.join(current_dir, annotation_dir))
    full_save_dir = os.path.abspath(os.path.join(current_dir, save_dir))

    annotation_files = glob.glob(full_annotation_dir + '*/*.json')

    list_filenames = []
    list_keypoints = []
    list_classname = []

    for file in annotation_files:
        dict_keypoints = get_json(file)

        filenames = list(dict_keypoints.keys())
        keypoints = list(dict_keypoints.values())

        classname = os.path.splitext(os.path.basename(file))[0]
        classname = [classname] * len(filenames)

        list_filenames = np.append(list_filenames, filenames)
        list_keypoints = np.append(list_keypoints, keypoints)
        list_classname = np.append(list_classname, classname)

    list_keypoints = list_keypoints.reshape(-1, 17, 2)
    angles = get_angles(list_keypoints)
    joint_distances = get_distances(list_keypoints)

    df = create_dataframe(
            keypoints=list_keypoints,
            filenames=list_filenames,
            classnames=list_classname,
            angles=angles,
            joint_distances=joint_distances
        )

    df.to_csv(full_save_dir, index=False)


if __name__ == '__main__':
    main()
