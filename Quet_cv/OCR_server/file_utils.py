# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Quet_cv.OCR_server.crop_images as crop_images
import Quet_cv.OCR_server.imgproc as imgproc
import matplotlib.image as mpimg
import concurrent.futures
from PIL import Image


# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files
def process_box_info(box, img, position):
    poly = np.array(box).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    xmin = min(poly[:, 0])
    xmax = max(poly[:, 0])
    ymin = min(poly[:, 1])
    ymax = max(poly[:, 1])
    h,w,d = img.shape 
    distance_ = w - xmax 
    w_line = xmax -xmin  
    if(xmin <= 50):
        xmin = 0
    else:
        if(w_line < 1/2*w):
            xmin -= 20
        else:
            xmin -= 50
    
    if(distance_ <= 50):
        xmax = w
    else:
        if(w_line < 1/2*w):
            xmax += 15
        else:    
            xmax += 50  
    if(ymin < 5):
        ymin = 0
    else:
        ymin -= 5
    height_distance = h - ymax
    if (height_distance < 5 ): 
        ymax = h
    else:
        ymax += 5 

    

    pts = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

    word = crop_images.crop(pts, img)

    return position, word
def get_image_line_Info(img, boxes):
    img = np.array(img)
    list_line = []

    # Sử dụng ThreadPoolExecutor để quản lý các luồng
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Tạo một future cho mỗi hộp trong danh sách hộp, gắn thêm thông tin vị trí
        futures = [executor.submit(process_box_info, box, img, position) for position, box in enumerate(boxes)]

        # Lấy kết quả từ các future khi chúng hoàn thành
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Sắp xếp lại kết quả theo vị trí
    results.sort(key=lambda x: x[0])

    # Thêm kết quả vào danh sách dòng
    for position, result in results:
        if result is not None:
            list_line.append(result)

    return list_line
def get_image_line_name_job(img,boxes,ocr_model):
    img = np.array(img)
    list_line = []
    print("len_box",len(boxes))
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        xmin = min(poly[:,0])
        xmax = max(poly[:,0])
        ymin = min(poly[:,1])
        ymax = max(poly[:,1])
        h,w,d = img.shape   
        ymax = max(poly[:, 1])
        h,w,d = img.shape 
        distance_ = w - xmax 
        w_line = xmax -xmin  
        if(xmin <= 50):
            xmin = 0
        else:
            if(w_line < 1/2*w):
                xmin -= 20
            else:
                xmin -= 50
        
        if(distance_ <= 50):
            xmax = w
        else:
            if(w_line < 1/2*w):
                xmax += 15
            else:    
                xmax += 50  
        if(ymin < 5):
            ymin = 0
        else:
            ymin -= 5
        height_distance = h - ymax
        if (height_distance < 5 ): 
            ymax = h
        else:
            ymax += 5 

        pts = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

        word = crop_images.crop(pts,img)
        
        if word is not None:
            image = Image.fromarray(word)

            image = image.convert("RGB")
            txt =   str(ocr_model.predict(image))
            print(txt)
            list_line.append(txt)

    return list_line

def process_box(box, img, position):
    poly = np.array(box).astype(np.int32).reshape((-1))
    poly = poly.reshape(-1, 2)
    xmin = min(poly[:, 0])
    xmax = max(poly[:, 0])
    ymin = min(poly[:, 1])
    ymax = max(poly[:, 1])
    h,w,d = img.shape 
    distance_ = w - xmax 
    w_line = xmax -xmin  
    if(xmin <= 50):
        xmin = 0
    else:
        if(w_line < 1/2*w):
            xmin -= 20
        else:
            xmin -= 50
    
    if(distance_ <= 50):
        xmax = w
    else:
        if(w_line < 1/2*w):
            xmax += 15
        else:    
            xmax += 50  
    if(ymin < 3):
        ymin = 0
    else:
        ymin -= 3
    height_distance = h - ymax
    if (height_distance < 3 ): 
        ymax = h
    else:
        ymax += 3 

    pts = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])

    word = crop_images.crop(pts, img)

    return position, word

def get_image_line(img, boxes):
    img = np.array(img)
    list_line = []

    # Sử dụng ThreadPoolExecutor để quản lý các luồng
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Tạo một future cho mỗi hộp trong danh sách hộp, gắn thêm thông tin vị trí
        futures = [executor.submit(process_box, box, img, position) for position, box in enumerate(boxes)]

        # Lấy kết quả từ các future khi chúng hoàn thành
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Sắp xếp lại kết quả theo vị trí
    results.sort(key=lambda x: x[0])

    # Thêm kết quả vào danh sách dòng
    for position, result in results:
        if result is not None:
            list_line.append(result)

    return list_line
