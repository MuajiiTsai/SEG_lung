from email.mime import image
import json
import os
import matplotlib
import cv2 as cv
# import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

matplotlib.use('Agg')
json_path = "./SEG_Train_Datasets/Train_Annotations/"
train_path = "./SEG_Train_Datasets/Train_Images/"

img_list = os.listdir(json_path)

for i in tqdm(img_list):
  filename = i[:-5]
  f = open(os.path.join(json_path, f'{filename}.json'))
  d = json.load(f)
  f.close()
  img = cv.imread(os.path.join(train_path, f'{filename}.jpg'))
  # plt.imshow(img)
  x_list = []
  y_list = []
  for j in range(len(d['shapes'][0]['points'])):
    tag = d['shapes'][0]['points'][j]
    x_list.append(tag[0])
    y_list.append(tag[1])

  plt.style.use('dark_background')
  plt.plot(x_list, y_list, color = 'white')
  plt.fill(x_list, y_list, color = 'white')
  plt.xlim(0, d['imageWidth'])  
  plt.ylim(0, d['imageHeight'])
  plt.axis('off')
  plt.savefig(f'./SEG_Train_Datasets/Annotation_Images/{filename}.jpg')
  plt.close()
  del img
print(len(os.listdir('./SEG_Train_Datasets/Annotation_Images')))
  
