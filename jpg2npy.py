import numpy as np
import os
import cv2 as cv
from tqdm import tqdm

datalist = os.listdir("./SEG_Train_Datasets/Train_Images")
datapath = "./SEG_Train_Datasets/"

if not os.path.isdir('./SEG_Train_Datasets/Annotation_npy/'):
  os.mkdir('./SEG_Train_Datasets/Annotation_npy/')
if not os.path.isdir('./SEG_Train_Datasets/Train_npy/'):
  os.mkdir('./SEG_Train_Datasets/Train_npy/')

for imgtype in enumerate(["Train_Images", "Annotation_Images"]):
  for i in tqdm(datalist):
    img = cv.imread(os.path.join(datapath, f'{imgtype[1]}/{i}'))
    if imgtype[0] == 0:
      np.save(f'./SEG_Train_Datasets/Train_npy/{i[:-4]}.npy', img)
    else:
      np.save(f'./SEG_Train_Datasets/Annotation_npy/{i[:-4]}.npy', img)
      # x = np.load(f'./SEG_Train_Datasets/Annotation_npy/{i[:-4]}.npy')
      # print(type(x[0][0][0]))
    # break
