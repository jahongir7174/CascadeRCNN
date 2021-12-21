import argparse
import copy
import os
import time

import mmcv
import mmdet
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist, get_dist_info, load_checkpoint, wrap_fp16_model
from mmcv.utils import get_git_hash, collect_env
from mmdet.apis import multi_gpu_test, single_gpu_test, set_random_seed, train_detector
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from utils.dataset import build_dataset


def train(args):
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = os.path.join('./weights',
                                os.path.splitext(os.path.basename(args.config))[0])
    cfg.gpu_ids = range(args.gpus)
    if args.distributed:
        # init distributed env first, since logger depends on the dist info.
        init_dist('pytorch', **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info_dict['MMDetection'] = mmdet.__version__ + '+' + get_git_hash()[:7]
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: True')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    logger.info(f'Set random seed to 0, deterministic: True')
    set_random_seed(0)
    cfg.seed = 0
    meta['seed'] = 0
    meta['exp_name'] = os.path.basename(args.config)

    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmdet_version=mmdet.__version__ + get_git_hash()[:7],
                                          CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets,
                   cfg, args.distributed,
                   True, timestamp, meta)


def test(args):
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.distributed:
        # init distributed env first, since logger depends on the dist info.
        init_dist('pytorch', **cfg.dist_params)
    rank, _ = get_dist_info()

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=args.distributed, shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.CLASSES = checkpoint['meta']['CLASSES']
    if args.distributed:
        model = MMDistributedDataParallel(model.cuda(),
                                          device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, gpu_collect=True)
        if get_dist_info()[0] == 0:
            dataset.format_results(outputs, jsonfile_prefix="./submission")
    else:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
        dataset.format_results(outputs, jsonfile_prefix="./submission")


def main():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--format-only',
                        default=True,
                        action='store_true',
                        help='Format the output results without perform evaluation. It is'
                             'useful when you want to format the result to a specific format and '
                             'submit it to the test server')
    parser.add_argument('--eval-options',
                        nargs='+',
                        default="jsonfile_prefix=./submission",
                        action=DictAction,
                        help='custom options for evaluation, the key-value pair in xxx=yyy '
                             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
