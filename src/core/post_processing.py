"""post-processing module after inference data."""
import numpy as np

from typing import List, Dict


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 192

def process_yolo(
        yolo_bboxes: np.ndarray,
        bbox_thr: float,
        class_id=0,
    ) -> List[Dict[str, np.ndarray]]:
    """Post processing bounding box output from yolo to mmpose format.

    Args:
        yolo_bboxes (np.ndarray): [N, 6] Output bounding boxes from yolo.
            [xmin, ymin, xmax, ymax, score, class_id]
        bbox_thr (float): bounding box threshold to be keep.
        class_id (int): Class id to select 0 default for person object.

    Returns:
        List N length contain dictionary with 'bbox' key and
            bouning box as value.
    """
    bboxes_all = yolo_bboxes.xyxy[0].detach().cpu().numpy()
    bboxes_person = bboxes_all[bboxes_all[:, 5] == class_id]
    bboxes_select = bboxes_person[bboxes_person[:, 4] > bbox_thr]
    bboxes_select = bboxes_select[:, :5]  # unselect class id at index 5
    person_results = []

    for bbox in bboxes_select:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def resize_keypoint(
        keypoints: np.ndarray,
        bounding_boxes: np.ndarray,
    ) -> np.ndarray:
    """Resizing keypoint in range of 256 (Height), 192 (Width).
    
    1. Cap bounding box and keypoint in an X, Y axis.
    2. find ratio to resize on X and Y.
    3. resize keypoint by computed ratio.

    Args:
        keypoints (np.ndarray): [N, 17, 2]
        bounding_boxes (np.ndarray): [N, 5]
            [xmin, ymin, xmax, ymax, score]

    Returns:
        List of resized keypoint [N, 17, 2].
    """
    xmin, ymin, xmax, ymax = bounding_boxes[:4]
    # 1 #
    keypoints_copy = keypoints.copy()
    keypoints_copy_X = keypoints_copy[:, 0] - xmin
    keypoints_copy_Y = keypoints_copy[:, 0] - ymin

    # 2 #
    ratio_x = (xmax - xmin) * IMAGE_WIDTH
    ratio_y = (ymax - ymin) * IMAGE_HEIGHT

    # 3 #
    keypoints_copy_x_resize = keypoints_copy_X * ratio_x
    keypoints_copy_y_resize = keypoints_copy_Y * ratio_y

    keypoints_resize = np.concatenate(
        [keypoints_copy_x_resize, keypoints_copy_y_resize], axis=1,
    )

    return keypoints_resize
