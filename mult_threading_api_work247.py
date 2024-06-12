from ultralytics import YOLO
import torch
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from Quet_cv.OCR_server.crop_box_img import YOLO_Detect
import Quet_cv.OCR_server.pipeline as pipeline
import Quet_cv.OCR_server.inference as inference
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
from Quet_cv.OCR_server.craft import CRAFT
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from Quet_cv.OCR_server.testCraft import copyStateDict
import urllib
from unidecode import unidecode
import multiprocessing
import Quet_cv.OCR_server.load_model as load_model
# from corrector_text import corrector
from underthesea import ner
# from utils.vietnamese_normalizer import no_accent_vietnamese
import subprocess 
import re 
# from DataManager.server import connect_elasticsearch
# client = connect_elasticsearch()

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
def extraxt_email(text):
    patterns = ['gmail.com', '[\w\.-]+@[\w\.-]+']
    for pattern in patterns:
        email = re.findall(pattern, text)
    if (len(email) != 0):
        return email[0]
    else:
        return ""

def extract_phone(text):
    patterns = ['[0-9]{10}', '[0-9]{5} [0-9]{5}',
                '[0-9]{4} [0-9]{3} [0-9]{3}',
                '[0-9]{4}.[0-9]{3}.[0-9]{3}',
                '[0-9]{3} [0-9]{3} [0-9]{4}',
                '[0-9]{3}.[0-9]{3}.[0-9]{4}',
                '[0-9]{4}-[0-9]{3}-[0-9]{3}',
                '[0-9]{3}-[0-9]{3}-[0-9]{4}']
    for pattern in patterns:
        if (re.findall(pattern, text)):
            return re.findall(pattern, text)[0]
    return ""
    

def extract_date_of_birth(text):
    text = text.replace('年 ', '年')
    text = text.replace(' 年', '年')
    text = text.replace('月 ', '月')
    text = text.replace(' 年', '年')
    text = text.replace('日 ', '日')
    text = text.replace(' 日', '日')
    patterns = ['[A-Za-z]+\s\d{1,2},\s\d{4}', '[A-Za-z]+\s\d{1,2}(?:st|nd|rd|th)?\s\d{4}', '\s\d{1,2} [A-Za-z]+ \s\d{4}',
                '\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}', '\d{2}\s[/.-]\s\d{2}\s[/.-]\s\d{4}',
                '[0-9]{4}年[0-9]{2}月[0-9]{2}日', '[A-Za-z]+\s\d{1,2}(?:St|Nd|Rd|Th)?\s\d{4}',
                '\d{2}[/.-]\d{2}[/.-]\d{4}', '\d{4}[/.-]\d{2}[/.-]\d{2}']
    birth = ''
    age = ''
    for pattern in patterns:
        if re.findall(pattern, text):
            birth = re.findall(pattern, text)[0]
            print('birth:', birth)
            break
    if re.findall('[0-9]{4}', birth):
        age = 2023 - int(re.findall('[0-9]{4}', birth)[0])
    return birth, age
def extract_gender(text):
    text = text.replace('Việt Nam', '')
    text = text.replace('việt nam', '')
    text = text.replace('Vietnam', '')
    text = text.replace('vietnam', '')
    patterns = ['Nam', 'Nữ', 'Male', 'Female', 'nam', 'nữ', 'NAM', 'NỮ']
    for pattern in patterns:
        if pattern in text:
            print('text:', pattern)
            return pattern
    else:
        return ""
import matplotlib
matplotlib.use('TkAgg')     
def extractText(image):
      
    boxes_list, label_list, detect_image, confs = det_box_model(image, output_path=None, return_result=True)

    cv2.imwrite(os.path.join(r"G:\RecommendSysterm\cropWord", 'detect_image.jpg'), detect_image)
    imgplot = plt.imshow(detect_image)
    plt.show()
    # plt.imshow(detect_image)
    # plt.show()
    cv ={}
    all_info =""
    sum_t = 0
    cropped_imgs =[]
    check_ = 0
    if(float(2) not in label_list or float(3) not in label_list or float(4) not in label_list ):
        check_ =1
    info = ""
    text_info =""
    cv_pdf_title = ""
    cv_pdf_name = ""
    all_info_box = ""
    for j, box in enumerate(boxes_list):       
        if(label_list[j]== float(2)):
            cropped_img = image[(int(box[1]) ):(int(box[3]) ) , (int(box[0]) ):(int(box[2]) ), :]
            list_crop_line = pipeline.crop_image_line_info(cropped_img,craft,args,refine_net)
            _ouput, all_info_box = inference.recog(list_crop_line,ocr_model_1)
            text_info = _ouput["title"] + " " + _ouput["text"] + " "
            info += text_info 
        elif(label_list[j]== float(3)):
            cropped_img = image[(int(box[1]) - 8):(int(box[3])+ 5 ) , (int(box[0]) - 8):(int(box[2]) +5  ), :]
            
            # cv["cv_pdf_title"] = _ouput["title"] + " " + _ouput["text"]
            text_title = pipeline.text_image_line_name_job(cropped_img,craft,args,refine_net, ocr_model_1)
            for i in range(len(text_title)) :    
                cv_pdf_title += text_title[i] + " "
            cv["cv_pdf_title"] = cv_pdf_title
            print(cv_pdf_title)
        elif(label_list[j]== float(4)):
            cropped_img = image[(int(box[1]) - 5):(int(box[3])+ 5 ) , (int(box[0]) - 5):(int(box[2]) +5  ), :]
            cv2.imwrite(os.path.join(r"G:\RecommendSysterm\cropWord", 'name.jpg'), cropped_img)

            # cv["cv_pdf_title"] = _ouput["title"] + " " + _ouput["text"]
            text_name = pipeline.text_image_line_name_job(cropped_img,craft,args,refine_net, ocr_model_1)
            # cv["cv_pdf_name"] = corrector(_ouput["title"] + " " + _ouput["text"])
            for i in range(len(text_name)):    
                cv_pdf_name += text_name[i] + " "
            # cv["cv_pdf_name"] = corrector(cv_pdf_name)
            cv["cv_pdf_name"] = cv_pdf_name
            # text_check = _ouput["title"] + " " + _ouput["text"]
            # Sử dụng NER để trích xuất thực thể từ văn bản
            entities = ner(cv_pdf_name)
            print(cv_pdf_name)

            for word, pos, tag, ner_ in entities:
                if ner_ == 'B-PER':
                    continue 
                elif ner_ == 'I-PER':
                    continue 
                else:
                    print("FAULT")
                    check_ = 1 


        elif(label_list[j]== float(0)):
            cv["cv_pdf_avatar"] = "avartar"
        else:
            cropped_img = image[(int(box[1])):(int(box[3])) , (int(box[0])):(int(box[2])), :]
            cropped_imgs.append(cropped_img)
            crst = time.time()
            list_crop_line = pipeline.crop_image_line(cropped_img,craft,args,refine_net)
            endcr = time.time()
            sum_t += endcr - crst
            _ouput, all_info_box = inference.recog(list_crop_line,ocr_model)
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
                        cv[str(list_box[0])] = _ouput["text"]
                        break
                    
                if found:
                    break
            print(all_info_box)
    
        all_info += all_info_box  
    email =  extraxt_email(info)
    phone =  extract_phone(info)
    date_of_birth, age  =  extract_date_of_birth(info)
    gender = extract_gender(info)
    cv["cv_pdf_email"] = email
    cv["cv_pdf_phone"] = phone
    cv["cv_pdf_age"] = age
    cv["cv_pdf_gender"] = gender
         
    cv["cv_pdf_thongtin"] =  info        
    # cv["cv_pdf_all"]  =  all_info + " " + cv_pdf_title + " " + corrector(cv_pdf_name)
    cv["cv_pdf_all"]  =  all_info + " " + cv_pdf_title + " " + cv_pdf_name
    # cv["cv_search_all"]  =  no_accent_vietnamese(all_info + " " + cv_pdf_title + " " + corrector(cv_pdf_name))
    # cv["cv_search_all"]  =  no_accent_vietnamese(all_info + " " + cv_pdf_title + " " + cv_pdf_name)     
    cv["cv_search_all"]  =  all_info + " " + cv_pdf_title + " " + cv_pdf_name
    return cv, check_, detect_image

def divide2image(image):
  # Kích thước ảnh ban đầu
  height, width, _ = image.shape

  # Kích thước của nửa trên và nửa dưới ảnh
  half_height = height // 2

  # Tạo hai nửa ảnh
  top_half = image[:half_height, :]
  bottom_half = image[half_height:, :]
  return top_half,bottom_half
def concatenate2dict(dict1,dict2):
    merged_dict = {}
    for key, value in dict1.items():
      if key in dict2:
          if key == "cv_pdf_all":
              merged_dict[key] = dict1[key] + " " + dict2[key]
          else:
              merged_dict[key] = dict1[key]
      else:
          merged_dict[key] = dict1[key]

    for key, value in dict2.items():
      if key not in merged_dict:
          merged_dict[key] = dict2[key]
    return merged_dict
def pdf2image(pdf_path, dpi):
    pages = convert_from_path(pdf_path, dpi)
    #pages = convert_from_path(pdf_path)
    image = np.vstack([np.asarray(page) for page in pages])
    return image
def docx2pdf(doc_path, path):

    subprocess.call(['soffice',
                  '--headless',
                 '--convert-to',
                 'pdf',
                 '--outdir',
                 path,
                 doc_path])
    return doc_path
def docx2pdf2image(pdf_path):
    pages = convert_from_path(pdf_path)
    image = np.vstack([np.asarray(page) for page in pages])
    return image

import time
class ErrorModel:
    def __init__(self, code, message):
        self.code = code
        self.message = message
app = Flask(__name__)


@app.route('/recognition', methods=['POST', 'GET'])
def recognition():
    try: 
        
        file_value = request.values['link_doc']
        id = request.values['id']
        site = request.values['site']
        index = 'tin_'+ site 
        extension = os.path.splitext(file_value)[1]

        print(f"id: {id} , link: {file_value} ")


        time_time = time.time()
        print("toi_day_het",round(time.time() - time_time,3))
        new_docx = urllib.request.urlretrieve(file_value,r"G:\RecommendSysterm\cropWord"+ "cv" + extension)
        print("toi_day_het",round(time.time() - time_time,3))
        ext_pdf = ['.pdf']
        ext_doc = ['.docx', '.doc']
        ext_img = ['.jpg', '.png']
        
        print(extension)
        
        if extension in ext_pdf:
            try:
                image = pdf2image(new_docx[0], 200)
                # print(new_docx[0])
                # print(image)
            except Exception as err:
                print('err:', err)

        elif extension in ext_doc: 
            path = r"G:\RecommendSysterm\cropWord"
            docx2pdf(new_docx[0], path)
            
            image = docx2pdf2image(new_docx[0].replace('.docx', '.pdf'))
        elif extension in ext_img:
            image = cv2.imread(new_docx[0])
        message = "Thanh Cong"    
        height, width, _ = image.shape
        if (width < height/3):
            image1,image2 = divide2image(image)
            
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
            out1,check1, detect_image_1 = extractText(image1)
            
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
            out2, check2,detect_image_2 = extractText(image2)
      
            out = concatenate2dict(out1,out2)
            file_name_link_image = r"G:\RecommendSysterm\cropWord\images_fault.txt"


            if (check1 == 1):
                # Mở tệp văn bản để ghi các liên kết
                with open(file_name_link_image, "a") as file:
                    file.write(file_value + "\n")
                # print("detect_image_1: ",detect_image_1.shape)
                # try:
                #     cv2.imwrite(os.path.join("/home/hungha/AI_365/timviec365_elasticsearch/Quet_cv/OCR_server/detect_fault",str(file_value)+ '_detect_image.jpg'), detect_image_1)
                #     print("detect_image_1: ",detect_image_1.shape)
                # except Exception as err:
                #     print('loi luu an', err)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            out,check,detect_image = extractText(image)
            file_name_link_image = r"G:\RecommendSysterm\cropWord\images_fault.txt"


            if (check == 1):
                # Mở tệp văn bản để ghi các liên kết
                with open(file_name_link_image, "a") as file:
                    file.write(file_value + "\n")
                # print("detect_image: ",detect_image.shape)
                # try:
                #     cv2.imwrite(os.path.join("/home/hungha/AI_365/timviec365_elasticsearch/Quet_cv/OCR_server/detect_fault",str(file_value)+ '_detect_image.jpg'), detect_image)
                #     print("detect_image: ",detect_image.shape)
                # except Exception as err:
                #     print('loi luu an', err)
                    
        print(out)
        # client.update(index=index, id = id , body = {"doc": out})

        
        print('1')
        return json.dumps({'status': 1,
            'error_code': 200,
            'message': message,
            'information': out})
    except Exception as err:
        message = 'Thông tin truyền lên không đầy đủ'
        print('err:', err)
        error = ErrorModel(200, message)
        print(error)

if __name__ == '__main__':
    det_box_model, ocr_model, craft, refine_net, args, ocr_model_1 = load_model.get_model()
    app.run(debug= False, host='0.0.0.0', port=8201)



