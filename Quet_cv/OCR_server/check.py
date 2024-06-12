import load_model 
import cv2 
import numpy as np
from PIL import Image
import pipeline
import matplotlib.pyplot as plt
det_box_model, ocr_model, craft, refine_net, args = load_model.get_model()

image = cv2.imread(r"C:\Users\ADMIN\Downloads\OCR_server\results\10.jpg")
# image = Image.fromarray(image)

# image = image.convert("RGB")
list_crop_line = pipeline.crop_image_line(image,craft,args,refine_net)
for img in list_crop_line:
    img = Image.fromarray(img)

    img = img.convert("RGB")
    plt.imshow(img)
    plt.show()
# out = ocr_model.predict(image)
# print(str(out))