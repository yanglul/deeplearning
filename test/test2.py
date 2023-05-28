import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
import numpy as np

device = select_device('0')
weights = ROOT / 'yolov5s.pt'
data = ROOT / 'data/coco128.yaml'
model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = (640, 640)  # check image size
imgsz = check_img_size(imgsz, s=stride)  # check image size
# Dataloader
bs = 1  # batch_size
def run():
    # Load model
    res = (0, 0)
    point = 204800

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt =  (Profile(), Profile(), Profile())

    im0s = cv2.imread('data/images/136.jpg')
    im = letterbox(im0s, imgsz, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, 0.25, 0.45, 0, False, max_det=1000)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    for i, det in enumerate(pred):  # per image

        im0 = im0s.copy()

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # imc = im0.copy() if False else im0  # for save_crop
        # annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xyxytemp = torch.tensor(xyxy).view(1, 4)

                templ = xyxytemp.numpy().tolist()
                print("templ", templ)
                temp = ((templ[0][0] + templ[0][2]) / 2, (templ[0][1] + templ[0][3]) / 2)
                print("temp", temp)
                d = (temp[0]-320)**2 + (temp[1]-320)**2
                if d<point:
                    point = d
                    res = (temp[0],temp[1])

            print("最近距离",point)
            print("最近距离坐标", res)

                # if True:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                #     print("*xywh", *xywh)


def forecast(im0s):
    res = (0, 0)
    point = 204800
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())


    im = letterbox(im0s, imgsz, stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, 0.25, 0.45, 0, False, max_det=1000)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


    # Process predictions
    for i, det in enumerate(pred):  # per image

        im0 = im0s.copy()
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # imc = im0.copy() if False else im0  # for save_crop
        # annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xyxytemp = torch.tensor(xyxy).view(1, 4)
                templ = xyxytemp.numpy().tolist()

                temp = ((templ[0][0] + templ[0][2]) / 2, (templ[0][1] + templ[0][3]) / 2)

                d = (temp[0] - 320) ** 2 + (temp[1] - 320) ** 2
                if d < point:
                    point = d
                    res = (temp[0], temp[1])

            print("最近距离", point)
            print("最近距离坐标", res)
            return res

if __name__ == '__main__':
    run()