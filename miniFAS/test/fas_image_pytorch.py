import os
import cv2
import warnings
from tqdm import tqdm
from miniFAS.function_model.fas import miniFAS_pytorch
from miniFAS.function_model.llie import is_low_light, lowlight_enhancement_pytorch
from src.anti_spoof_predict import AntiSpoofPredict
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
model_test = AntiSpoofPredict(0)
dataset ="datasets/Test/data_real_lowlight"
   
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0

    label = 'real'
    cnt_lowlight = 0
    images = os.listdir(dataset)
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        # print("img shape: ",img.shape)
        threshold = 100
        if is_low_light(img,threshold):
            img = lowlight_enhancement_pytorch(img)
            cnt_lowlight += 1
        prediction = miniFAS_pytorch(img)
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
    print("count low light: ", cnt_lowlight)