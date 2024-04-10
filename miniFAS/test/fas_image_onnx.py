import os
import cv2
import warnings
import time
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('miniFAS')
from function_model.llie import LowLightEnhancer
from function_model.fas import FaceAntiSpoofing
warnings.filterwarnings('ignore')

dataset = "miniFAS/datasets/Test/dark_face_dataset"
model_1 = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
threshold = 100
scale_factor = 12
model_llie = 'miniFAS/model_onnx/ZeroDCE++.onnx'
save_path_llie = 'miniFAS/datasets/Test/dark_face_dataset_enhancement'
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    count_none_face = 0
    count_llie = 0
    fas_model1 = FaceAntiSpoofing(model_1)
    fas_model2 = FaceAntiSpoofing(model_2)
    lowlight_enhancer = LowLightEnhancer(scale_factor=12, model_onnx=model_llie)
    dir_label = os.listdir(dataset)
    for label in dir_label:
        save_path_llie_result = os.path.join(save_path_llie, label)
        dir_img = os.path.join(dataset, label)
        images = os.listdir(dir_img)
        print("len folder " + label +": ",len(images))
        TIME_START = time.time()
        for image in tqdm(images):
            prediction = np.zeros((1, 3))
            img_path = os.path.join(dir_img, image)
            img = cv2.imread(img_path) #BGR

            if lowlight_enhancer.is_lowlight(img,threshold):
                img = lowlight_enhancer.enhance(img[:,:,::-1]) #RGB
                # result_path = os.path.join(save_path_llie_result, image)
                # cv2.imwrite(result_path, img)
                img = img[:,:,::-1]
                count_llie += 1

            if fas_model1.is_face(img) is None:
                count_none_face += 1
                continue
            else: 
                prediction = fas_model1.predict(img) + fas_model2.predict(img)
                output = np.argmax(prediction)
        
                if output != 1 and label == 'fake':
                    tp += 1
                elif output == 1 and label == 'fake':
                    fn += 1
                elif output != 1 and label == 'real':
                    fp += 1
                elif output == 1 and label == 'real':
                    tn += 1
            
    print("tp:", tp)
    print("fp:", fp)
    print("fn:", fn)
    print("tn:", tn)
    print("count low light: ", count_llie)
    print("count none face: ", count_none_face)
    print("time: ", time.time() - TIME_START)