import os


def get_parent_folder(raw_path: str) -> str:
    """
        Get class name (parent directories of this file) from full path.

        Args:
            path (str): Raw full path name.
        
        Returns:
            A string class name (parent directories of this file) of full path.
            example:

                'projects/folder1/folder2/item.py' -> 'folder2'
    """

    return os.path.basename(os.path.dirname(raw_path))

def get_save_pathname(raw_path: str, classname: str, target_path: str) -> str:
    """
        Get full path name to be save after being resize.

        Args:
            raw_path (str): Raw path name.
            classname (str): parent directories of raw path.
            target_path (str): Target path name to be save.
        
        Returns:
            A string full path name to be save.
            example:

                'target_path/classname/item_resize.jpg'
    """
    filename_without_extension, extension = os.path.splitext(os.path.basename(raw_path))
    filename_save= filename_without_extension + '_resize' + extension
    save_path = os.path.join(target_path, classname, filename_save)
    
    return save_path
