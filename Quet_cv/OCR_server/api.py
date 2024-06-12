from ultralytics import YOLO
import torch
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from crop_box_img import YOLO_Detect
import pipeline
import inference
import glob
import numpy as np
from pdf2image import convert_from_path
from flask import Flask,request
import json
from PIL import Image
import uuid
import io
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from craft import CRAFT
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from test import copyStateDict
import urllib
from unidecode import unidecode
import multiprocessing
import load_model

list_box_name =["cv_pdf_avatar","cv_pdf_thongtin","cv_pdf_name","cv_pdf_title","cv_pdf_chungchi","cv_pdf_giaithuong","cv_pdf_duanthamgia","cv_pdf_sothich","cv_pdf_kynang","cv_pdf_hoatdong","cv_pdf_kinhnghiem","cv_pdf_all","cv_pdf_muctieu","cv_pdf_thongtinthem","cv_pdf_hocvan"]

map_label = {0: 'avatar',
             1: 'block',
             2: 'infor',
             3: 'job_title',
             4: 'name'}
def chuyen_cau_khong_dau_chu_thuong(cau):
    # Chuyển đổi văn bản có dấu thành văn bản không dấu
    cau_khong_dau = unidecode(cau)
    
    # Chuyển câu thành chữ thường
    cau_khong_dau_chu_thuong = cau_khong_dau.lower()
    
    return cau_khong_dau_chu_thuong
def process_cropped_image(cropped_img):
    list_crop_line = pipeline.crop_image_line(cropped_img,craft,args,refine_net)
    return list_crop_line
    
def extractText(image):
    

    
    boxes_list, label_list, detect_image, confs = det_box_model(image, output_path=None, return_result=True)

    cv2.imwrite(os.path.join("/home/hungha/AI_365/timviec365_elasticsearch/Quet_cv/OCR_server/cropWord/", 'detect_image.jpg'), detect_image)
    cv ={}
    all_info =""
    sum_t = 0
    cropped_imgs =[]
    for j, box in enumerate(boxes_list):
        # print(box)
        # print(1)
        cropped_img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        # cv2.imwrite(os.path.join("C:/Users/ADMIN/Downloads/OCR_server/results/",str(j) + '.jpg'), cropped_img)
        cropped_imgs.append(cropped_img)
        # print(2)  
        crst = time.time()
        list_crop_line = pipeline.crop_image_line(cropped_img,craft,args,refine_net)
        # for img in list_crop_line:
        #     img = Image.fromarray(img)

        #     img = img.convert("RGB")
        #     plt.imshow(img)
        #     plt.show()
        endcr = time.time()
        sum_t += endcr - crst
        # print(len(list_crop_line))
        # print(3)
        # paths = glob.glob("/home/hungha/AI_365/nga/OCR_server/cropWord/*.jpg")

        _ouput, all_info_box = inference.recog(list_crop_line,ocr_model)
        print(label_list[j])
        print(_ouput)
        
        if(label_list[j]== float(2)):
            cv["cv_pdf_thongtin"] = _ouput
        elif(label_list[j]== float(3)):
            cv["cv_pdf_title"] = _ouput
        elif(label_list[j]== float(4)):
            cv["cv_pdf_name"] = _ouput
        elif(label_list[j]== float(0)):
            cv["cv_pdf_avatar"] = _ouput
        else:
            sentence_title  = str(_ouput["title"])
            sentence_title = chuyen_cau_khong_dau_chu_thuong(sentence_title)
            # cv_pdf_chungchi            
            list_chungchi =["chungchi","chung chi","chung ch","hung chi","chungch"]
            # cv_pdf_giaithuong
            list_giaithuong =["cv_pdf_giaithuong","giaithuong","giai thuong","iai thuong", "giai thuon","giaithuon"]
            # cv_pdf_duanthamgia
            list_cv_pdf_duanthamgia =["cv_pdf_duanthamgia","du an tham gia","du an thamgia","u an tham gia","du an tham gi","du an thamgi","duan tham gia"]
            # "cv_pdf_sothich",
            list_cv_pdf_sothich =["cv_pdf_sothich","so thich","sothich","so thic","o thich","sothic"]
            # "cv_pdf_kynang",
            list_kynang = ["cv_pdf_kynang","ky nang","kynang","kynan","ky nan","y nang"]
            # "cv_pdf_hoatdong",
            list_hoatdong = ["cv_pdf_hoatdong","hoatdong","hoat dong","hoatdon","hoat don","oat dong"]
            # "cv_pdf_kinhnghiem",
            list_kinhnghiem = ["cv_pdf_kinhnghiem","kinh nghiem","kinhnghiem","kinhnghie","kinh nghie","inh nghiem"]
            
            # "cv_pdf_muctieu",
            list_muctieu =["cv_pdf_muctieu","muctieu","muc tieu","muctie","muc tie","uc tieu"]
            # "cv_pdf_thongtinthem",
            list_thongtinthem = ["cv_pdf_thongtinthem","thongtinthem","thong tinthem","thongtin them","thong tin them","thong tinthe","hong tin them","thong tin the"]
            # "cv_pdf_hocvan"
            list_hocvan =["cv_pdf_hocvan","hocvan","hoc van","hocva","hoc va","oc van"]
            list_boxs = [list_chungchi,list_giaithuong,list_cv_pdf_duanthamgia,list_cv_pdf_sothich,list_kynang,list_hoatdong,list_kinhnghiem,list_muctieu,list_thongtinthem,list_hocvan]
            found = False
            for list_box in list_boxs :
                for i in range(1,len(list_box)):
                    if list_box[i]  in sentence_title:
                        found = True
                        # print(f"Tìm thấy từ '{list_box[i]}' trong câu.")
                        cv[str(list_box[0])] = _ouput
                        break
                    
                if found:
                    break
            
    
        all_info += all_info_box             
    print("---------")   
    print(sum_t)     
    # "cv_pdf_all",  
    cv["cv_pdf_all"]  =  all_info 
        # print(_ouput)
        # print(j)
        # cv["box"+str(j)] = _ouput
        # shutil.rmtree("/home/hungha/AI_365/nga/OCR_server/cropWord")
    return cv

        

import time
app = Flask(__name__)


@app.route('/recognition', methods=['POST', 'GET'])
def recognition():
    
    file_value = request.values['link_doc']
    time_time = time.time()
    print("toi_day_het",round(time.time() - time_time,3))
    new_docx = urllib.request.urlretrieve(file_value,"/home/hungha/AI_365/timviec365_elasticsearch/Quet_cv/OCR_server/cropWord/"+ str(uuid.uuid4()) + '.png')
    print("toi_day_het",round(time.time() - time_time,3))

    message = "Thanh Cong"    
    image = cv2.imread(new_docx[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = extractText(image)
    print(out)
    print("tong toi_day_het",round(time.time() - time_time,3))
    # shutil.rmtree(folder_image)
    
    
    return json.dumps({'status': 1,
        'error_code': 200,
        'message': message,
        'information': out})

if __name__ == '__main__':
    det_box_model, ocr_model, craft, refine_net, args = load_model.get_model()
    app.run(debug= True, host='43.239.223.4', port=8101)



