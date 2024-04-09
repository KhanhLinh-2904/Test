import os
import cv2
import warnings
from tqdm import tqdm
import numpy as np
import sys

sys.path.append("miniFAS")
from function_model.fas import FaceAntiSpoofing

warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

# model_test = AntiSpoofPredict(0)
filePath = "miniFAS/datasets/Test/dark_face_dataset/img.png"
model_1 = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"

if __name__ == "__main__":

    fas_model1 = FaceAntiSpoofing(model_1)
    fas_model2 = FaceAntiSpoofing(model_2)

    prediction = np.zeros((1, 3))
    img = cv2.imread(filePath)

    if fas_model1.is_face(img) is None or fas_model2.is_face(img) is None:
        print("There is no face here!")
    else:
        prediction = fas_model1.predict(img) + fas_model2.predict(img)
        label = np.argmax(prediction)
        print(label)
        if label == 1:
            print("Real")
        else:
            print("Fake")
