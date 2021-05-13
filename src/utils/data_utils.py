"""Contain function doing about data."""
import json
import os

import numpy as np
from typing import Dict


def get_json(filename: str) -> Dict[str, np.ndarray]:
    """Open filename.json and return as a dictionary object.

    Args:
        filename (str): Json file name to be used.

    Returns:
        Dictionary loaded from json file.
    """
    with open(filename, 'r') as json_file:
        dict_ = json.load(json_file)

    return dict_


def get_parent_folder(raw_path: str) -> str:
    """Get class name (parent directories of this file) from full raw_path.

    Args:
        raw_path (str): Raw full path name.

    Returns:
        A string class name (parent directories of this file) of full path.
        example:

        'projects/folder1/folder2/item.py' -> 'folder2'
    """

    return os.path.basename(os.path.dirname(raw_path))


def get_save_pathname(
        raw_path: str,
        classname: str,
        target_path: str,
    ) -> str:
    """Get full path name to be save after being resize.

    Args:
        raw_path (str): Raw path name.
        classname (str): parent directories of raw path.
        target_path (str): Target path name to be save.

    Returns:
        A string full path name to be save.
        example:

        'target_path/classname/item_resize.jpg'
    """
    filename_without_ext, ext = os.path.splitext(os.path.basename(raw_path))
    filename_save = filename_without_ext + '_resize' + ext
    save_path = os.path.join(target_path, classname, filename_save)

    return save_path
