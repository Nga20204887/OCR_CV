from ultralytics import YOLO
import torch
import cv2
import os
import matplotlib.pyplot as plt
import shutil

map_label = {0: 'avatar',
             1: 'block',
             2: 'infor',
             3: 'job_title',
             4: 'name'}


class YOLO_Detect:
    def __init__(self, weight_path=None, model_name=None):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weight_path)
        self.model.to(self.device)

    def __call__(self, image, return_result=False, output_path=None):
        result = self.model(image, conf=0.4, verbose=False, nms=True, iou=0.2)
        boxes_list = result[0].boxes.data[:, :4]
        label_idx = result[0].boxes.data[:, 5]
        # labels = [map_label[int(i)] for i in label_idx]
        box = result[0].boxes
        conf = box.conf

        detect_image = result[0].plot()
        return boxes_list, label_idx, detect_image, conf

# if not os.path.exists('C:/Users/ADMIN/Documents/OCR_server/test'):
#     os.mkdir('C:/Users/ADMIN/Documents/OCR_server/test')
# output_path = 'C:/Users/ADMIN/Documents/OCR_server/test'

# weight_path = 'C:/Users/ADMIN/Downloads/best.pt'
# det_model = YOLO_Detect(weight_path=weight_path)
# image = cv2.imread('C:/Users/ADMIN/Downloads/cv_1695274406_1371188.png')

# boxes_list, label_list, detect_image, confs = det_model(image, output_path=output_path, return_result=True)

# # cv2.imwrite(os.path.join(output_path, 'detect_image.jpg'), detect_image)
# boxes_list, label_list, detect_image, confs = det_model(image, output_path=output_path, return_result=True)
# for j, box in enumerate(boxes_list):
#     # print(box)
#     cropped_img = image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
#     cv2.imwrite(os.path.join(output_path, 'img'+str(j)+'.jpg'), cropped_img)
# # shutil.rmtree(output_path)
