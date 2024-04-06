import os
import cv2
import warnings
import time
from tqdm import tqdm

from function_model.fas import miniFAS_onnx
from function_model.llie import is_low_light, lowlight_enhancement_onnx
from src.anti_spoof_predict import AntiSpoofPredict
warnings.filterwarnings('ignore')

model_test = AntiSpoofPredict(0)
dataset ="datasets/Test/dark_face_dataset/real"
 
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0

    label = 'real'
    images = os.listdir(dataset)
    count = 0
    print("len folder: ",len(images))
    TIME_START = time.time()
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        threshold = 100
        if is_low_light(img,threshold):
            img = lowlight_enhancement_onnx(img)
            count += 1
        prediction = miniFAS_onnx(img)
        if prediction == "none":
            print("There is no face here!")
            continue
    
        if prediction == "fake" and label == 'fake':
            tp += 1
        elif prediction == 'real' and label == 'fake':
            fn += 1
        elif prediction == 'fake' and label == 'real':
            fp += 1
        elif prediction == 'real' and label == 'real':
            tn += 1
            
    print("tp:", tp)
    print("fp:", fp)
    print("fn:", fn)
    print("tn:", tn)
    print("count low light: ", count)
    print("time: ", time.time() - TIME_START)