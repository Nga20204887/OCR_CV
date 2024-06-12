""" Deskews file after getting skew angle """
import optparse
import numpy as np
import matplotlib.pyplot as plt

from Quet_cv.OCR_server.skew_detect import SkewDetect
from skimage import io
from skimage.transform import rotate
import Quet_cv.OCR_server.load_model as load_model


class Deskew:

    def __init__(self, input, r_angle):

        self.input = input
        # self.display_image = display_image
        # self.output = output
        self.r_angle = r_angle
        self.skew_obj = SkewDetect(self.input)

    def deskew(self):

        # img = io.imread(self.input)
        img = self.input
        res = self.skew_obj.process_single_file()
        angle = res['Estimated Angle']
        print(angle)

        if angle >= 0 and angle <= 90:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -45 and angle < 0:
            rot_angle = angle - 90 + self.r_angle
        if angle >= -90 and angle < -45:
            rot_angle = 90 + angle + self.r_angle

        rotated = rotate(img, rot_angle, resize=True)
        
        return rotated

 
# import cv2 
# import time
# import load_model 
# from PIL import Image
# import pipeline
# det_box_model, ocr_model, craft, refine_net, args = load_model.get_model()
# st = time.time()
# img ="C:/Users/ADMIN/Downloads/b05ecc7e63f17a9f1795256634ff944f.jpg"
# image =  io.imread(img, as_gray=True)
# check_skew = SkewDetect(image)
# res = check_skew.process_single_file()
# angle = res['Estimated Angle']
# print(angle)
# deskew = Deskew(image,0)

# out = deskew.deskew()
# out_uint8 = (out * 255).astype(np.uint8)

# # Tạo ảnh từ mảng uint8
# image = Image.fromarray(out_uint8)
# img_3d = image.convert("L")
# image = img_3d.convert("RGB")
# img_array = np.array(image)

# list_crop_line = pipeline.crop_image_line_(img_array,craft,args,refine_net,ocr_model,"ok",1 )
# plt.imshow(image)
# plt.show()
# txt = str(ocr_model.predict(image))
# img = cv2.imread("G:/OCR_server/data_crawl_/image_0box_2position_0.jpg")
# img = Image.fromarray(img)

# print("text2:" ,str(ocr_model.predict(img)) )
# print("txt",txt)
# end = time.time()
# print(end - st)
# plt.imshow(out)
# plt.show()