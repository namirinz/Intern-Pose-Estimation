import argparse
import time

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmpose.datasets import build_dataloader as pose_build_dataloader
from mmpose.datasets import build_dataset as pose_build_dataset
from mmdet.datasets import build_dataloader as det_build_dataloader
from mmdet.datasets import build_dataset as det_build_dataset

from mmpose.models import build_posenet
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose benchmark a recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--mmdet-bbox', action='store_true', help='(for TopDown only) Using mmdet model to extract bbox first')
    parser.add_argument(
        '--checkpoint',  help='')
    parser.add_argument(
        '--det-config', help='detection config file if TopDown else blank')
    parser.add_argument(
        '--log-interval', default=10, type=int, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args

def inference_BottomUp(pose_model, data_loader_pose, num_warmup, pure_inf_time, log_interval):
    print("Inference Pose Model only")
    for i, data_pose in enumerate(data_loader_pose):
        print(data_pose)
        break
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            pose_model(return_loss=False, **data_pose)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                its = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done item [{i + 1:<3}],  {its:.2f} items / s')

        if (i + 1) == 2000:
            pure_inf_time += elapsed
            its = (i + 1 - num_warmup) / pure_inf_time
            return its

def inference_TopDown(pose_model, det_model, data_loader_pose, data_loader_det, num_warmup, pure_inf_time, log_interval):
    print("Inference Detection & Pose Model")
    for i, (data_pose, data_det) in enumerate(zip(data_loader_pose, data_loader_det)):

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            det_model(return_loss=False, rescale=True, **data_det)
            pose_model(return_loss=False, **data_pose)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                its = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done item [{i + 1:<3}],  {its:.2f} items / s')

        if (i + 1) == 2000:
            pure_inf_time += elapsed
            its = (i + 1 - num_warmup) / pure_inf_time
            return its

def main():
    args = parse_args()

    pose_cfg = Config.fromfile(args.config)
    if args.mmdet_bbox:
        if pose_cfg.model.type == 'BottomUp':
            raise ValueError('BottomUp Model doesn\'t need to use mmdet for getting bounding box. Don\'t specify --mmdet-bbox when using BottomUp Model')
        elif pose_cfg.model.type == 'TopDown' and args.det_config is None:
            raise ValueError('TopDown Model need to specify mmdet model if using --mmdet-bbox. Specify with --det-config argument')
        det_cfg = Config.fromfile(args.det_config)
        det_cfg.data.test.test_mode = True

    # set cudnn_benchmark
    if pose_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset_pose = pose_build_dataset(pose_cfg.data.val)
    data_loader_pose = pose_build_dataloader(
        dataset_pose,
        samples_per_gpu=1,
        workers_per_gpu=pose_cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    if args.mmdet_bbox:
        det_cfg.model.train_cfg = None
        detector_model = build_detector(det_cfg.model, test_cfg=det_cfg.get('test_cfg'))
        load_checkpoint(detector_model, args.checkpoint, map_location='cpu')

        dataset_det = det_build_dataset(det_cfg.data.test)
        data_loader_det = det_build_dataloader(
            dataset_det,
            samples_per_gpu=1,
            workers_per_gpu=det_cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )
        if args.fuse_conv_bn :
            detector_model = fuse_conv_bn(detector_model)
        detector_model = MMDataParallel(detector_model, device_ids=[0])
        detector_model.eval()

    pose_model = build_posenet(pose_cfg.model)
    fp16_cfg = pose_cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(pose_model)
    if args.fuse_conv_bn:
        pose_model = fuse_conv_bn(pose_model)
    pose_model = MMDataParallel(pose_model, device_ids=[0])

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with total batch and take the average
    if args.mmdet_bbox:
        its = inference_TopDown(pose_model, detector_model, data_loader_pose, data_loader_det,
                    num_warmup, pure_inf_time, args.log_interval)
    else:
        its = inference_BottomUp(pose_model, data_loader_pose, num_warmup, pure_inf_time, args.log_interval)

    print(f'Overall average: {its:.2f} items / s')
    print(f'Total time: {pure_inf_time:.2f} s')


if __name__ == '__main__':
    main()
