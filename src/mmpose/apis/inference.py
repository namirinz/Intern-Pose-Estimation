
import torch
import numpy as np
import cv2

from mmpose.datasets.pipelines import Compose
from mmcv.parallel import collate

def process_yolo(det_results, score_thr=0.3, cat_id=0):
    bboxes_xyxy = det_results.xyxy
    bboxes_xyxy = bboxes_xyxy[0].detach().cpu().numpy()
    bboxes_xyxy = bboxes_xyxy[(bboxes_xyxy[:, -1] == cat_id) & (bboxes_xyxy[:, -2] > score_thr)]
    
    bboxes_xywh = bboxes_xyxy.copy()

    bboxes_xywh[:, 2] = bboxes_xywh[:, 2]-bboxes_xywh[:, 0]
    bboxes_xywh[:, 3] = bboxes_xywh[:, 3]-bboxes_xywh[:, 1]

    return bboxes_xyxy, bboxes_xywh

def _box2cs(cfg, box):
    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale

class LoadImage:
    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        results['image_file'] = ''
        if self.color_type == 'color' and self.channel_order == 'rgb':
            img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)

        results['img'] = img
        return results

def inf_single_pose(model, config, img, bboxes_xywh, bboxes_xyxy):
    device = next(model.parameters()).device

    # build the data pipeline
    channel_order = config.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + config.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    
    batch_data = []
    for bbox in bboxes_xywh:
        center, scale = _box2cs(config, bbox)

        # prepare data
        data = {
        'img_or_path': img,
        'center': center,
        'scale': scale,
        'bbox_score': bbox[4],
        'bbox_id': 0,
        'dataset': 'TopDownCocoDataset',
        'joints_3d': np.zeros((config.data_cfg.num_joints, 3), dtype=np.float32),
        'joints_3d_visible': np.zeros((config.data_cfg.num_joints, 3), dtype=np.float32),
        'rotation': 0,
        'ann_info': {
            'image_size': config.data_cfg['image_size'],
            'num_joints': config.data_cfg['num_joints'],
            'flip_pairs': flip_pairs
            }
        }

        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        batch_data['img'] = batch_data['img'].to(device)
    batch_data['img_metas'] = [ img_metas[0] for img_metas in batch_data['img_metas'].data]
    
    # forward the model
    with torch.no_grad():
        results = model(img=batch_data['img'], img_metas=batch_data['img_metas'], return_loss=False, return_heatmap=False)
        
    return [{'bbox': bboxes_xyxy[i, :-1], 'keypoints': results['preds'][i]} for i in range(len(results['boxes']))]

def vis_det_result(img, results, classes, score_thr=0.3, font=cv2.FONT_HERSHEY_PLAIN, thickness=6, text_color=(0, 255 ,0), text_color_bg=(0, 0, 255), font_scale=2, font_thickness=3):
    img_ = img.copy()
    for idx, class_ in enumerate(results):
        text = classes[idx]
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        for bbox in class_:
            xmin, ymin, xmax, ymax, score = bbox
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
            if score < score_thr: continue
            # xmin, ymin, xmax, ymax
            img_ = cv2.rectangle(img_, (xmin, ymin), (xmax, ymax), text_color_bg, thickness)
            img_ = cv2.rectangle(img_, (xmin, ymin), (xmin+text_width, ymin+text_height), text_color_bg, -1)
            img_ = cv2.putText(img_, text, (xmin, ymin + text_height + font_scale - 1), font, font_scale, text_color, font_thickness)
    return img_

def vis_pose_result(model,
                    img,
                    result,
                    kpt_score_thr=0.3,
                    dataset='TopDownCocoDataset',
                    radius=4,
                    thickness=1,
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])


    if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                   'TopDownOCHumanDataset'):
        # show the results
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
        ]]

    elif dataset == 'TopDownCocoWholeBodyDataset':
        # show the results
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [16, 18],
                    [16, 19], [16, 20], [17, 21], [17, 22], [17, 23], [92, 93],
                    [93, 94], [94, 95], [95, 96], [92, 97], [97, 98], [98, 99],
                    [99, 100], [92, 101], [101, 102], [102, 103], [103, 104],
                    [92, 105], [105, 106], [106, 107], [107, 108], [92, 109],
                    [109, 110], [110, 111], [111, 112], [113, 114], [114, 115],
                    [115, 116], [116, 117], [113, 118], [118, 119], [119, 120],
                    [120, 121], [113, 122], [122, 123], [123, 124], [124, 125],
                    [113, 126], [126, 127], [127, 128], [128, 129], [113, 130],
                    [130, 131], [131, 132], [132, 133]]

        pose_limb_color = palette[
            [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16] +
            [16, 16, 16, 16, 16, 16] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
        pose_kpt_color = palette[
            [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
            [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]

    elif dataset == 'TopDownAicDataset':
        skeleton = [[3, 2], [2, 1], [1, 14], [14, 4], [4, 5], [5, 6], [9, 8],
                    [8, 7], [7, 10], [10, 11], [11, 12], [13, 14], [1, 7],
                    [4, 10]]

        pose_limb_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 0, 7, 7
        ]]
        pose_kpt_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 0, 0
        ]]

    elif dataset == 'TopDownMpiiDataset':
        skeleton = [[1, 2], [2, 3], [3, 7], [7, 4], [4, 5], [5, 6], [7, 8],
                    [8, 9], [9, 10], [9, 13], [13, 12], [12, 11], [9, 14],
                    [14, 15], [15, 16]]

        pose_limb_color = palette[[
            16, 16, 16, 16, 16, 16, 7, 7, 0, 9, 9, 9, 9, 9, 9
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 16, 7, 7, 0, 0, 9, 9, 9, 9, 9, 9
        ]]

    elif dataset == 'TopDownMpiiTrbDataset':
        skeleton = [[13, 14], [14, 1], [14, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                    [1, 7], [2, 8], [7, 8], [7, 9], [8, 10], [9, 11], [10, 12],
                    [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26],
                    [27, 28], [29, 30], [31, 32], [33, 34], [35, 36], [37, 38],
                    [39, 40]]

        pose_limb_color = palette[[16] * 14 + [19] * 13]
        pose_kpt_color = palette[[16] * 14 + [0] * 26]

    elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                     'PanopticDataset'):
        skeleton = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8],
                    [8, 9], [1, 10], [10, 11], [11, 12], [12, 13], [1, 14],
                    [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20],
                    [20, 21]]

        pose_limb_color = palette[[
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ]]

    elif dataset == 'InterHand2DDataset':
        skeleton = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10],
                    [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18],
                    [18, 19], [19, 20], [4, 21], [8, 21], [12, 21], [16, 21],
                    [20, 21]]

        pose_limb_color = palette[[
            0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12, 16
        ]]
        pose_kpt_color = palette[[
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16,
            0
        ]]

    elif dataset == 'Face300WDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 68]
        kpt_score_thr = 0

    elif dataset == 'FaceAFLWDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 19]
        kpt_score_thr = 0

    elif dataset == 'FaceCOFWDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 29]
        kpt_score_thr = 0

    elif dataset == 'FaceWFLWDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 98]
        kpt_score_thr = 0

    else:
        raise NotImplementedError()

    img = model.show_result(
        img,
        result,
        skeleton,
        radius=radius,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        show=show,
        thickness=thickness,
        out_file=out_file)

    return img