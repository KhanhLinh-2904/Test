import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import torchvision 
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision import  utils
import sys
sys.path.append('miniFAS')
from function_model.llie import is_low_light, lowlight_enhancement_onnx
filePath = 'datasets/Test/dark_face_dataset/fake'	
savePath = ''
saveModel = True
if __name__ == '__main__':

    with torch.no_grad():
        file_list = os.listdir(filePath)
        sum_time = 0
        threshold = 100
        for file_name in file_list:
            print("file_name:",file_name)
            path_to_image = os.path.join(filePath, file_name)
            print("path_to_image:",path_to_image)
            img = cv2.imread(path_to_image)
            if is_low_light(img,threshold):
                lowlight_enhancement_onnx(img, file_name, saveModel, savePath)
            else:
                result_path = os.path.join(savePath, file_name)
                img.save(result_path)
       
        