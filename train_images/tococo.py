import json
import numpy as np
from pycocotools import mask
from skimage import measure
from PIL import Image
import os
import cv2

annotation = []
images = []

pths = [f for f in os.listdir('.') if os.path.isdir(f)]

ann_id = 0
img_id = 0
for pth in pths:
    fulpth = os.path.abspath(os.getcwd())+'/'+pth+'/masks/'
    for f in os.listdir(fulpth):
        im = cv2.imread(fulpth+f)
        if(im is None):
            print(fulpth+f)
            continue
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = [contour.reshape(contour.shape[0], -1).astype('int')
                    for contour in contours]

        min_x = np.min([np.min(contour[:, 0]) for contour in contours])
        min_y = np.min([np.min(contour[: 1]) for contour in contours])
        max_x = np.max([np.max(contour[:, 0]) for contour in contours])
        max_y = np.max([np.max(contour[:, 1]) for contour in contours])
        w = max_x - min_x
        h = max_y - min_y
        # Segmentation
        segmentation_contours = [
            list(map(int, list(contour.flatten()))) for contour in contours]

        annotation.append({
            'image_id': img_id,
            'iscrowd': 0,
            'bbox': [int(min_x), int(min_y), int(w), int(h)],
            'segmentation': segmentation_contours,
            'id': ann_id,
            "category_id": 1,
        })
        ann_id += 1
    images.append({
        "file_name": pth+'/images/'+pth+'.png',
        "id": img_id,
        "height": 1000,
        "width": 1000
    })
    img_id += 1

dmp = {"annotations": annotation, "images": images, "categories": [
    {
        "name": "nuclei",
        "id": 1
    }
]}

with open("train.json", "w") as f:
    json.dump(dmp, f)
