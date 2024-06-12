"""Modify to Remove Argument Parser"""

"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import Quet_cv.OCR_server.craft_utils as craft_utils
import Quet_cv.OCR_server.imgproc as imgproc
import Quet_cv.OCR_server.file_utils as file_utils
import json
import zipfile

from Quet_cv.OCR_server.craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, args, refine_net=None):
    t0 = time.time()
    # resize
    # cứ biết là lấy ra kích thước ảnh mới , tỉ lệ với chiều rộng chiều cao ảnh, kích thước bản đồ nhiệt
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)

    #đây chính là lấy tỉ lệ với chiều rộng và chiều cao ảnh còn là gì thì xuống xem
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing Bước tiền xử lý
    x = imgproc.normalizeMeanVariance(img_resized)
    '''
    chú ý rằng tensor trong pytorch bằng với array trong numpy
    trong 2 lệnh tiếp theo 
    đầu tiên là thay đổi thứ tự và dòng tiếp tăng số chiều của nó lên
    mục đích chưa rõ
    https://github.com/pytorch/pytorch/issues/44541
    '''
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

    if cuda:
        x = x.cpu()

    # forward pass
    # từ net(x) lấy ra y và feature nữa, mục đích chưa rõ
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    #những tham số truyền vào này này đều là tham số default từ file với pipeline
    # mục đích là lấy ra boxes, polys, det_scores mới
    # quá trình xử lý chưa rõ
    # Post-processing gọi hàm từ file craft_utils
    # boxes để lấy tọa độ, polys có vẻ là dương tự nhưng mà dưới dạng weight hoặc cách implement khác
    #
    boxes, polys, det_scores = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment gọi hàm từ file craft_utils
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]
    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text, det_scores