[CascadeRCNN](https://arxiv.org/abs/1906.09756) implementation using PyTorch

### Install

* `pip install mmcv-full==1.5.1`
* `pip install mmdet==2.24.1`

### Train

* `bash ./main.sh ./nets/exp01.py $ --train` for training, `$` is number of GPUs

### Results

|   Detector    | Backbone |      Neck      | LR Schedule | Box mAP | Mask mAP | Config |
|:-------------:|:--------:|:--------------:|:-----------:|--------:|---------:|-------:|
| Cascade R-CNN |  Swin-T  |      FPN       |     1x      |       - |        - |  exp01 |
| Cascade R-CNN |  Swin-T  |      FPN       |     3x      |       - |        - |  exp02 |
| Cascade R-CNN |  Swin-T  |      FPN       |     3x      |       - |        - |  exp03 |
| Cascade R-CNN |  Swin-T  |      FPN       |     3x      |       - |        - |  exp04 |
| Cascade R-CNN |  Swin-T  |     PAFPN      |     3x      |       - |        - |  exp05 |
| Cascade R-CNN |  Swin-T  | PAFPN + DyHead |     3x      |       - |        - |  exp06 |

### TODO

* [x] [exp01](./nets/exp01.py), default [Cascade R-CNN](https://arxiv.org/abs/1906.09756) detector
* [x] [exp02](./nets/exp02.py), added [MOSAIC](https://arxiv.org/abs/2004.10934) and [MixUp](https://arxiv.org/abs/1710.09412)
* [x] [exp03](./nets/exp03.py), added [GN](https://arxiv.org/abs/1803.08494) and [WS](https://arxiv.org/abs/1903.10520)
* [x] [exp04](./nets/exp04.py), added [PolyLoss](https://arxiv.org/abs/2204.12511)
* [x] [exp05](./nets/exp05.py), added [PAFPN](https://arxiv.org/abs/1803.01534)
* [x] [exp06](./nets/exp06.py), added [DyHead](https://arxiv.org/abs/2106.08322)

### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmdetection