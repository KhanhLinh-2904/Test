import os
import cv2
import warnings
from tqdm import tqdm
import sys
sys.path.append('miniFAS')
from function_model.fas import miniFAS_pytorch
from src.anti_spoof_predict import AntiSpoofPredict
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt

model_test = AntiSpoofPredict(0)
filePath = 'miniFAS/datasets/Test/dark_face_dataset/dark_printouts_img_0 (1).png'	


if __name__ == "__main__":
   
    img = cv2.imread(filePath)
    prediction = miniFAS_pytorch(img)
    if prediction == "none":
        print("There is no face here!")
    elif prediction == "fake":
        print("Fake")
    elif prediction == 'real':
        print("Real")
   