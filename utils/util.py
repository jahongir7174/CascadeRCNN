import glob
import math
import os
import random

import cv2
import numpy
from mmdet.datasets.pipelines.transforms import PIPELINES


def resize(image, image_size):
    h, w = image.shape[:2]
    ratio = image_size / max(h, w)
    if ratio != 1:
        shape = (int(w * ratio), int(h * ratio))
        image = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
    return image, image.shape[:2]


def xy2wh(x):
    y = numpy.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyn2xy(x, w, h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:, 0] = w * x[:, 0] + pad_w  # top left x
    y[:, 1] = h * x[:, 1] + pad_h  # top left y
    return y


def whn2xy(x, w, h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def mask2box(mask, w, h):
    x, y = mask.T
    inside = (x >= 0) & (y >= 0) & (x <= w) & (y <= h)
    x, y, = x[inside], y[inside]
    if any(x):
        return numpy.array([x.min(), y.min(), x.max(), y.max()]), x, y
    else:
        return numpy.zeros((1, 4)), x, y


def box_ioa(box1, box2, eps=1E-7):
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    area1 = (numpy.minimum(b1_x2, b2_x2) - numpy.maximum(b1_x1, b2_x1)).clip(0)
    area2 = (numpy.minimum(b1_y2, b2_y2) - numpy.maximum(b1_y1, b2_y1)).clip(0)
    inter_area = area1 * area2

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def masks2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xy2wh(numpy.array(boxes))


def resample_masks(masks, n=1000):
    for i, s in enumerate(masks):
        x = numpy.linspace(0, len(s) - 1, n)
        xp = numpy.arange(len(s))
        mask = [numpy.interp(x, xp, s[:, i]) for i in range(2)]
        masks[i] = numpy.concatenate(mask).reshape(2, -1).T
    return masks


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = numpy.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def copy_paste(image, boxes, masks, p=0.):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177
    n = len(masks)
    if p and n:
        h, w, c = image.shape
        img = numpy.zeros(image.shape, numpy.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = boxes[j], masks[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = box_ioa(box, boxes[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                boxes = numpy.concatenate((boxes, [[l[0], *box]]), 0)
                masks.append(numpy.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(img, [masks[j].astype(numpy.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=image, src2=img)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        image[i] = result[i]

    return image, boxes, masks


def random_perspective(image, boxes=(), masks=(),
                       degrees=0, translate=.1, scale=.5,
                       shear=0, perspective=0., border=(0, 0)):
    h = image.shape[0] + border[0] * 2
    w = image.shape[1] + border[1] * 2

    # Center
    c_gain = numpy.eye(3)
    c_gain[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    c_gain[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    p_gain = numpy.eye(3)
    p_gain[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    p_gain[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    r_gain = numpy.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    r_gain[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    s_gain = numpy.eye(3)
    s_gain[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    s_gain[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    t_gain = numpy.eye(3)
    t_gain[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w  # x translation (pixels)
    t_gain[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h  # y translation (pixels)

    # Combined rotation matrix
    matrix = t_gain @ s_gain @ r_gain @ p_gain @ c_gain  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():
        if perspective:
            image = cv2.warpPerspective(image, matrix, dsize=(w, h), borderValue=(0, 0, 0))
        else:  # affine
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    n = len(boxes)
    if n:
        new_masks = []
        new_boxes = numpy.zeros((n, 4))
        for i, mask in enumerate(resample_masks(masks)):
            xy = numpy.ones((len(mask), 3))
            xy[:, :2] = mask
            xy = xy @ matrix.T
            xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]

            # clip
            new_boxes[i], x, y = mask2box(xy, w, h)
            new_masks.append([x, y])

        # filter candidates
        candidates = box_candidates(boxes[:, 1:5].T * s, new_boxes.T, area_thr=0.01)
        boxes = boxes[candidates]
        boxes[:, 1:5] = new_boxes[candidates]
        masks = []
        for candidate, new_mask in zip(candidates, new_masks):
            if candidate:
                masks.append(new_mask)
    return image, boxes, masks


@PIPELINES.register_module()
class TSCopyPaste:
    def __init__(self, data_dir='../Dataset/Ins2021'):
        self.data_dir = data_dir

        self.crop_image_dir = 'c_images'
        self.crop_label_dir = 'c_labels'

        self.src_names_0 = glob.glob(f'{data_dir}/{self.crop_image_dir}/0/*.png')
        self.src_names_1 = glob.glob(f'{data_dir}/{self.crop_image_dir}/1/*.png')

        with open(f'{data_dir}/{self.crop_label_dir}/0.txt') as f:
            self.labels = {}
            for line in f.readlines():
                line = line.rstrip().split(' ')
                self.labels[line[0]] = line[1:]
        with open(f'{data_dir}/{self.crop_label_dir}/1.txt') as f:
            for line in f.readlines():
                line = line.rstrip().split(' ')
                self.labels[line[0]] = line[1:]

    def paste(self, img):
        gt_label = []
        gt_masks = []
        gt_boxes = []

        dst_h, dst_w = img.shape[:2]
        num_0 = numpy.random.randint(5, 15)
        num_1 = numpy.random.randint(5, 15)
        y_c_list = numpy.random.randint(dst_h // 2 - 256, dst_h // 2 + 256, num_0 + num_1)
        x_c_list = numpy.random.randint(256, dst_w - 256, num_0 + num_1)
        src_names = numpy.random.choice(self.src_names_0, num_0).tolist()
        src_names.extend(numpy.random.choice(self.src_names_1, num_1).tolist())

        mask_list = []
        poly_list = []
        src_img_list = []
        src_name_list = []
        for src_name in src_names:
            poly = []
            label = self.labels[os.path.basename(src_name)]
            src_img = cv2.imread(src_name)
            for i in range(0, len(label), 2):
                poly.append([int(label[i]), int(label[i + 1])])
            src_mask = numpy.zeros(src_img.shape, src_img.dtype)
            cv2.fillPoly(src_mask, [numpy.array(poly)], (255, 255, 255))
            mask_list.append(src_mask)
            poly_list.append(poly)
            src_img_list.append(src_img)
            src_name_list.append(src_name)
        for i, (x_c, y_c) in enumerate(zip(x_c_list, y_c_list)):
            dst_poly = []
            for p in poly_list[i]:
                dst_poly.append([int(p[0] + x_c), int(p[1] + y_c)])
            dst_mask = numpy.zeros(img.shape, img.dtype)
            cv2.fillPoly(dst_mask, [numpy.array(dst_poly, int)], (255, 255, 255))
            x_min, y_min, w, h = cv2.boundingRect(numpy.array([dst_poly], int))
            gt_boxes.append([x_min, y_min, x_min + w, y_min + h])
            src = src_img_list[i].copy()
            h, w = src.shape[:2]
            mask = mask_list[i].copy()
            img[dst_mask > 0] = 0
            img[y_c:y_c + h, x_c:x_c + w] += src * (mask > 0)
            if 'human' in os.path.basename(src_name_list[i]):
                gt_label.append(0)
            else:
                gt_label.append(1)
            dst_point = []
            for p in dst_poly:
                dst_point.append(p[0])
                dst_point.append(p[1])
            gt_masks.append([dst_point])
        return img, gt_label, gt_boxes, gt_masks

    def __call__(self, results):
        img = results['img']
        img, label, boxes, masks = self.paste(img)

        masks.extend(results['ann_info']['masks'])
        label.extend(results['ann_info']['labels'].tolist())
        boxes.extend(results['ann_info']['bboxes'].tolist())

        results['img'] = img
        results['ann_info']['labels'] = numpy.array(label, numpy.int64)
        results['ann_info']['bboxes'] = numpy.array(boxes, numpy.float32)
        results['ann_info']['masks'] = masks
        return results
