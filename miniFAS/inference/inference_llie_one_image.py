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
filePath = '/home/user/low_light_enhancement/Zero-DCE++/data/Test_Part2/1_4.jpg'	
image_name = 'img.png'
savePath = 'miniFAS/datasets/Test/dark_face_dataset/'
saveModel = True
if __name__ == '__main__':

    with torch.no_grad():
        threshold = 100
        print("file_name:",filePath)
        img = cv2.imread(filePath) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if is_low_light(img,threshold):
            lowlight_enhancement_onnx(img, image_name, saveModel, savePath)
        else:
            result_path = os.path.join(savePath, image_name)
            img.save(result_path)
       
