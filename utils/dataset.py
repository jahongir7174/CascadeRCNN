import copy
import random

import numpy
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose

from utils import util


def load_mosaic(self, index):
    boxes4, masks4 = [], []
    size = numpy.random.choice(self.image_sizes)
    border = [-size // 2, -size // 2]
    indexes4 = [index] + random.choices(range(self.num_samples), k=3)
    yc, xc = [int(random.uniform(-x, 2 * size + x)) for x in border]
    numpy.random.shuffle(indexes4)
    results4 = [copy.deepcopy(self.dataset[index]) for index in indexes4]

    shapes = [x['img_shape'][:2] for x in results4]
    image4 = numpy.full((2 * size, 2 * size, 3), 0, numpy.uint8)

    for i, (results, shape) in enumerate(zip(results4, shapes)):
        image, (h, w) = util.resize(results['img'], size)

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, size * 2), min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        masks = []
        label = numpy.array(results['ann_info']['labels'])
        for mask in results['ann_info']['masks']:
            mask = [j for i in mask for j in i]
            mask = numpy.array(mask).reshape(-1, 2)
            masks.append(mask / numpy.array([shape[1], shape[0]]))
        masks = [x for x in masks]
        try:
            boxes = (label.reshape(-1, 1), util.masks2boxes(masks))
            boxes = numpy.concatenate(boxes, axis=1)
        except IndexError:
            return None
        if boxes.size:
            boxes[:, 1:] = util.whn2xy(boxes[:, 1:], w, h, pad_w, pad_h)
            masks = [util.xyn2xy(x, w, h, pad_w, pad_h) for x in masks]
        boxes4.append(boxes)
        masks4.extend(masks)
    # concatenate & clip
    boxes4 = numpy.concatenate(boxes4, 0)
    for i, box4 in enumerate(boxes4[:, 1:]):
        if i % 2 == 0:
            numpy.clip(box4, 0, 2 * size, out=box4)
        else:
            numpy.clip(box4, 0, 2 * size, out=box4)
    for mask4 in masks4:
        numpy.clip(mask4[:, 0:1], 0, 2 * size, out=mask4[:, 0:1])
        numpy.clip(mask4[:, 1:2], 0, 2 * size, out=mask4[:, 1:2])
    image4, boxes4, masks4 = util.copy_paste(image4, boxes4, masks4, p=0.0)
    image4, boxes4, masks4 = util.random_perspective(image4, boxes4, masks4, border=border)

    label = []
    boxes = []
    masks = []
    for box4, mask4 in zip(boxes4, masks4):
        mask = []
        for x, y in zip(mask4[0], mask4[1]):
            mask.append(x)
            mask.append(y)
        masks.append([mask])
        label.append(box4[0])
        boxes.append(box4[1:5])
    if len(boxes) and len(masks) and len(label):
        label = numpy.array(label, dtype=numpy.int64)
        boxes = numpy.array(boxes, dtype=numpy.float32)

        results = results4[0]

        results['ann_info']['masks'] = masks
        results['ann_info']['labels'] = label
        results['ann_info']['bboxes'] = boxes
        results['img'] = image4
        results['img_info']['height'] = image4.shape[0]
        results['img_info']['width'] = image4.shape[1]
        results['img_shape'] = image4.shape
        results['ori_shape'] = image4.shape
        return self.pipeline(results)
    else:
        return None


@DATASETS.register_module()
class MOSAICDataset:
    def __init__(self, dataset, image_size, pipeline):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.pipeline = Compose(pipeline)
        if hasattr(self.dataset, 'flag'):
            self.flag = numpy.zeros(len(dataset), dtype=numpy.uint8)
        self.image_sizes = image_size
        self.num_samples = len(dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        while True:
            data = load_mosaic(self, index)
            if data is None:
                index = self.dataset._rand_another(index)
                continue
            return data


def build_dataset(cfg, default_args=None):
    if cfg['type'] == 'MOSAICDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        return MOSAICDataset(**cp_cfg)
    else:
        return build_from_cfg(cfg, DATASETS, default_args)
