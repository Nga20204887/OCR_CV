import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from Quet_cv.OCR_server.testCraft import copyStateDict
from PIL import Image
import cv2
from skimage import io
import numpy as np
import Quet_cv.OCR_server.craft_utils as craft_utils
import Quet_cv.OCR_server.testCraft as testCraft
import Quet_cv.OCR_server.imgproc as imgproc
import Quet_cv.OCR_server.file_utils as file_utils
import json
import zipfile
import pandas as pd
from Quet_cv.OCR_server.craft import CRAFT
from collections import OrderedDict
from Quet_cv.OCR_server.skew_detect import SkewDetect
import matplotlib.pyplot as plt
from Quet_cv.OCR_server.deskew import Deskew

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
def crop_image_line(image,net,args,refine_net):

    net.eval()
    
    bboxes, polys, score_text, det_scores = testCraft.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cpu, args.poly, args, refine_net)

    list_image_line = file_utils.get_image_line(image,polys)
    
    return list_image_line
def crop_image_line_info(image,net,args,refine_net):

    net.eval()
    
    bboxes, polys, score_text, det_scores = testCraft.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cpu, args.poly, args, refine_net)

    list_image_line = file_utils.get_image_line_Info(image,polys)
    
    return list_image_line
def text_image_line_name_job(image,net,args,refine_net,ocr_model):
    
    check_skew = SkewDetect(image)
    res = check_skew.process_single_file()
    angle = res['Estimated Angle']
    if((angle**2 > 30**2  and angle**2 < 75**2) or ( angle >= 75 and  angle <= 90)):
        deskew = Deskew(image,0)
        out = deskew.deskew()

        out_uint8 = (out * 255).astype(np.uint8)

        # Tạo ảnh từ mảng uint8
        image = Image.fromarray(out_uint8)
        img_3d = image.convert("L")
        image = img_3d.convert("RGB")
        image = np.array(image)
    # cv2.imwrite(os.path.join("/home/hungha/AI_365/timviec365_elasticsearch/Quet_cv/OCR_server/results/",'skew'+ '.jpg'), image)


    net.eval()
    
    bboxes, polys, score_text, det_scores = testCraft.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cpu, args.poly, args, refine_net)

    list_image_line = file_utils.get_image_line_name_job(image,polys,ocr_model)
    
    return list_image_line
