import os
import glob
import cv2

from src.utils import get_save_pathname
from src.utils import get_parent_folder
from typing import List, Tuple


def get_filenames() -> Tuple[List[str], List[str]]:
    """
        Get full path image file name, directories and path to be save after being resize.

        Returns:
            A tuple with 2 list of string there are:
                1. list of image filenames from data/raw folder.
                2. list of path to be save after being resize.
            example:

                (
                 ['folder/data/raw/action/images/standing/1_standing.jpg'],
                 ['folder/data/raw/action/images/standing/1_standing_resize.jpg']
                )
    """

    current_dir = os.path.dirname(__file__)
    
    raw_dir = "../data/raw/action/images"
    processed_dir = "../data/processed/action/images"
    
    full_raw_dir = os.path.abspath(os.path.join(current_dir, raw_dir))
    full_raw_dir_filenames = glob.glob(full_raw_dir+'/*/*')
    full_target_dir = os.path.abspath(os.path.join(current_dir, processed_dir))

    save_directories = []

    for file in full_raw_dir_filenames:
        parent_folder  = get_parent_folder(file)
        save_path = get_save_pathname(file, parent_folder, full_target_dir)

        save_directories.append(save_path)

    return (full_raw_dir_filenames, save_directories)

def main():
    raw_filenames, target_filenames = get_filenames()

    for raw_file, target_file in zip(raw_filenames, target_filenames):
        image_raw = cv2.imread(raw_file, cv2.IMREAD_COLOR)
        image_resize = cv2.resize(image_raw, [192,256]) # [Width, Height]
        cv2.imwrite(target_file, image_resize)

if __name__ == '__main__':
    main()
    