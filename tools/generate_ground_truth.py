""" Generate ground truth .csv from annotation."""
import glob
import os

import pandas as pd
import numpy as np

from src.core.feature_engineer import get_angle
from src.core.feature_engineer import get_distances
from src.core.feature_engineer import keypoint_to_vectors

from src.utils import get_json
from src.utils.const import KEYPOINT_COLS
from src.utils.const import ANGLE_COLS
from src.utils.const import ANGLE_JOINTS
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
    ) -> pd.DataFrame:
    """Create ground truth dataframe.
    
    Args:
        keypoints (np.ndarray): [N, 17, 2] Keypoint in x, y coordinate.
        filenames (List[str]): List of file name.
        classnames (List[str]): List of class name.
    Returns:
        Dataframe contain ... features. with label and file names.
    """
    KEYPOINT_COLS_XY = make_xy_column(KEYPOINT_COLS)
    
    df = pd.DataFrame(keypoints, columns=KEYPOINT_COLS_XY)
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
    
    df = create_dataframe(
            keypoints=list_keypoints, filenames=list_filenames,
            classname=list_classname
        )


if __name__ == '__main__':
    main()
