import os
import shutil
import cv2
import cv2 as cv
import numpy as np
import argparse
import warnings
import time
from tqdm import tqdm
import torch
import sys
sys.path.append('miniFAS')
from function_model.fas import load_fas_onnx
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')
model_test = AntiSpoofPredict(0)
dataset ="miniFAS/datasets/Test/dark_face_dataset/fake"

model_1 = 'miniFAS/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'miniFAS/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model = model_2

def cal_mean_abs_diff(prediction_py, prediction_onnx):
    
    diff = cv2.absdiff(prediction_py, prediction_onnx)
    # print("diff: ", diff)
    mean_absolute_difference = np.mean(diff)
    return mean_absolute_difference


def predict_two_model_pytorch(image):
    model_test = AntiSpoofPredict(0)
    image_cropper =  CropImage()
    image_bbox, conf = model_test.get_bbox(image)
    if conf < 0.7:
        return "none"
    prediction = np.zeros((1,3))
    for model in [ model_1]:
        model_name = model.split("/")[-1]
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)[0]

        prediction += model_test.predict(img, model)
    return prediction
    
def predict_two_model_onnx(image):
    prediction = np.zeros((1, 3))
    for model in [ model_2]:
        flag = load_fas_onnx(model, image)
        if type(flag) == type("none"):
            return "none"
        else:
            prediction += flag
    return prediction
    
if __name__ == "__main__":
    
    mean_absolute_differences = []
    label = 'fake'
    images = os.listdir(dataset)
    print("len folder: ", len(images))
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
       
        prediction_py = predict_two_model_pytorch(img)
        prediction_onnx = predict_two_model_onnx(img)
        if type(prediction_py) == type("none") or type(prediction_onnx) == type("none"):
            continue
        maf_value = cal_mean_abs_diff(prediction_py,prediction_onnx)
        # print(type(maf_value))
        mean_absolute_differences.append(maf_value)
        # print(mean_absolute_differences)
    overall_mean_absolute_difference = np.mean(mean_absolute_differences)
    print(f"Overall Mean Absolute Difference: {overall_mean_absolute_difference:.f}")
