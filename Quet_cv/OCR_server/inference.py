from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import time
import csv
import glob
import concurrent.futures
import threading
from PIL import Image
def process_image(image_ndaarray, detector, position):
    image = Image.fromarray(image_ndaarray)
    image = image.convert("RGB")
    result = str(detector.predict(image))
    return position, result

def recog_line(list_images_ndarray, detector):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image, detector, position) for position, image in enumerate(list_images_ndarray)]
    
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda x: x[0])  # Sắp xếp theo vị trí
    return [result[1] for result in results]

def recog(list_images_ndarray,detector):
    # box_text_lines = []
    st = time.time()
    results = recog_line(list_images_ndarray, detector)
   
    end = time.time()
    # print(end - st)
    text = ""
    title =""
    if len(list_images_ndarray) > 1:
        print('>1')
        for i, result in enumerate(results):
            if i == 0:
                title = result
            else:
                text += result + " "
        box ={
            "title" : title,
            "text" : text
        }
        all_info_box = title + " " + text
    elif(len(list_images_ndarray)==1):
        print('=1')
        box ={
            "title" : str(results[0]),
            "text" : text
        }
        all_info_box = str(results[0]) + " " + text
    else:
        print('khác')
        box ={"title":"avatar",
              "text" : text}
        all_info_box = ""
    
    return box, all_info_box
# def process_image(index,image_ndaarray, detector, results):
#     image = Image.fromarray(image_ndaarray)
#     image = image.convert("RGB")
#     # plt.imshow(image)
#     # plt.show()
#     result = str(detector.predict(image))
#     # print(result)
#     results.append((index,result))
    

# def recog_line(list_images_ndarray, detector):
#     # Danh sách kết quả
#     box_text_lines = []


#     # Danh sách luồng
#     threads = []

#     # Danh sách kết quả từ các luồng
#     results = []
#     # st = time.time()
#     # Tạo và khởi chạy các luồng
#     for index,image_ndaarray in enumerate(list_images_ndarray):
#         thread = threading.Thread(target=process_image, args=(index,image_ndaarray, detector, results))
#         thread.start()
#         threads.append(thread)
#     for thread in threads:
#         thread.join()
#     # end = time.time()
#     # print(end - st)

#     # Không cần chờ cho đến khi tất cả các luồng hoàn thành

#     # Xử lý kết quả ở đây (đảm bảo rằng bạn đã xử lý đúng đắn các kết quả từ các luồng)

#     return results

# def recog(list_images_ndarray,detector):
#     # box_text_lines = []
#     st = time.time()
#     results = recog_line(list_images_ndarray, detector)
#     # for result in results:
#     #     print(result)
#     # for image_ndaarray in list_images_ndarray:
#     #     image = Image.fromarray(image_ndaarray)

#     #     image = image.convert("RGB")
#     #     # print(path)
#     #     # n = Image.open(path)
#     #     # print(str(detector.predict(n)))
#     #     box_text_lines.append(str(detector.predict(image)))
#     end = time.time()
#     print(end - st)
#     text = ""
#     title =""
#     if len(list_images_ndarray) > 1:
#         for i, result in results:
#             if i == 0:
#                 title = result
#             else:
#                 text += result + " "
#         box ={
#             "title" : title,
#             "text" : text
#         }
#         all_info_box = title + " " + text
#     elif(len(list_images_ndarray)==1):
#         box ={
#             "title" : str(results[0][1]),
#             "text" : text
#         }
#         all_info_box = str(results[0][1]) + " " + text
#     else:
#         box ={"avatar":"avatar"}
#         all_info_box = ""
    
#     return box, all_info_box
        
# paths = glob.glob("C:/Users/ADMIN/Documents/OCR_server/cropWord/*.jpg")
# cv_img = []
# for path in paths:
# # for i in range(len(path)):
#     print(path)
#     n = Image.open(path)
#     print(str(detector.predict(n)))
#     cv_img.append(str(detector.predict(n)))
# def process_image(image_ndaarray, detector):
#     image = Image.fromarray(image_ndaarray)
#     image = image.convert("RGB")
#     result = str(detector.predict(image))
#     return result

# def recog_line(list_images_ndarray, detector):
#     # Sử dụng ThreadPoolExecutor để quản lý các luồng
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_image, image, detector) for image in list_images_ndarray]
    
#     # Lấy kết quả từ các future khi chúng hoàn thành
#     results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
#     return results
# def recog(list_images_ndarray,detector):
#     # box_text_lines = []
#     st = time.time()
#     results = recog_line(list_images_ndarray, detector)
   
#     end = time.time()
#     print(end - st)
#     text = ""
#     title =""
#     if len(list_images_ndarray) > 1:
#         for i, result in enumerate(results):
#             if i == 0:
#                 title = result
#             else:
#                 text += result + " "
#         box ={
#             "title" : title,
#             "text" : text
#         }
#         all_info_box = title + " " + text
#     elif(len(list_images_ndarray)==1):
#         box ={
#             "title" : str(results[0]),
#             "text" : text
#         }
#         all_info_box = str(results[0]) + " " + text
#     else:
#         box ={"avatar":"avatar"}
#         all_info_box = ""
    
#     return box, all_info_box

        
