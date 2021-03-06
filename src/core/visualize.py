import cv2

import numpy as np
from matplotlib import pyplot as plt
from src.utils.const import KEYPOINT_PAIRS


BLUE_COLOR = (0, 0, 255)
ORANGE_COLOR = (255, 140, 0)
GREEN_COLOR = (0, 188, 0)


def draw_bbox(
        bboxes: np.ndarray,
        image: np.ndarray,
        format='xyxy',
    ) -> np.ndarray:
    """Draw bounding box over image.

    Args:
        bboxes (np.ndarray): [N, 5] or [N, 4] bounding box of object.
        image (np.ndarray): [H, W, 3] Image with BGR format.
        format (str): format of bounding box 'xyxy' stand for
            xmin, ymin, xmax, ymax and 'xywh' stand for
            xmin, ymin, width, height.
    """
    xmin, ymin, xmax, ymax = bboxes[:4]


def draw_keypoint(
        keypoints: np.ndarray,
        image: np.ndarray,
        radius=4,
        thickness=1,
        show=False,
        save_path=None,
    ) -> np.ndarray:
    """Visualize plotted keypoint and choose to plot or save plotted image.

    Args:
        keypoints (np.ndarray): [17, 2] 17 points by coco format with x, y.
        image (np.ndarray): [H, W, 3] Image with BGR format.
        radius (int): Radius of circle keypoint to be plot.
        thickness (int): Line thickness between 2 circle keypoint.
        show (bool): Whether to show a plotted image or not.
        save_path (str): Path to save.

    Returns:
        A numpy array plotted image with BGR format.
    """

    image_plot = image.copy()

    color_pairs = [
        BLUE_COLOR, BLUE_COLOR, BLUE_COLOR,
        BLUE_COLOR, BLUE_COLOR, ORANGE_COLOR,
        ORANGE_COLOR, ORANGE_COLOR, ORANGE_COLOR,
        ORANGE_COLOR, ORANGE_COLOR, GREEN_COLOR,
        GREEN_COLOR, GREEN_COLOR, GREEN_COLOR,
        GREEN_COLOR, GREEN_COLOR,
    ]

    for pair in KEYPOINT_PAIRS:

        # Select keypoint by index from pair
        first_point = keypoints[pair[0]].astype(np.int32)
        first_point = tuple(first_point)

        second_point = keypoints[pair[1]].astype(np.int32)
        second_point = tuple(second_point)

        first_point_color = color_pairs[pair[0]]
        second_point_color = color_pairs[pair[1]]
        line_color = first_point_color

        # Plotting first point
        image_plot = cv2.circle(
            img=image_plot, center=first_point,
            color=first_point_color, radius=radius, thickness=-1,
        )

        # Plotting second point
        image_plot = cv2.circle(
            img=image_plot, center=second_point,
            color=second_point_color, radius=radius, thickness=-1,
        )

        # Plotting line between 2 points
        image_plot = cv2.line(
            img=image_plot, pt1=first_point, pt2=second_point,
            color=line_color, thickness=thickness,
        )

    if show:
        plt.imshow(cv2.cvtColor(image_plot, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    if save_path is not None:
        cv2.imwrite(save_path, image_plot)

    return image_plot
