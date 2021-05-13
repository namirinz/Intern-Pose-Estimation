""" Generate ground truth .csv from annotation."""
import glob
import json
import os

import pandas as pd
import numpy as np

from src.core.feature_engineer import get_angle
from src.core.feature_engineer import get_distances
from src.core.feature_engineer import keypoint_to_vectors
from src.utils import get_json
from typing import Dict, List


KEYPOINT_COLS = [
    "nose", "left_eye", "right_eye", "left_ear",
    "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

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
    
    list_keypoints = list_keypoints.reshape(-1, 17 * 2)
    
    KEYPOINT_COLS_XY = make_xy_column(KEYPOINT_COLS)
    df = pd.DataFrame(list_keypoints, columns=KEYPOINT_COLS_XY)
    df['filename'] = list_filenames
    df['class_name'] = list_classname
    df.to_csv(full_save_dir, index=False)


if __name__ == '__main__':
    main()
