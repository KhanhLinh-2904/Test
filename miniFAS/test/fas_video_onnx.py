import os
import cv2
import warnings
from tqdm import tqdm
import torch
import numpy as np
import sys
sys.path.append('miniFAS')

from function_model.fas import FaceAntiSpoofing
from function_model.llie import LowLightEnhancer

warnings.filterwarnings('ignore')
# Initialize GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset ="miniFAS/datasets/Test/low-light-video"
model_1 = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
model_llie = 'miniFAS/model_onnx/ZeroDCE++.onnx'
label = 'real'
threshold = 100
scale_factor = 12

if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    non_face = 0
    fas_model1 = FaceAntiSpoofing(model_1)
    fas_model2 = FaceAntiSpoofing(model_2)
    lowlight_enhancer = LowLightEnhancer(scale_factor=12, model_onnx=model_llie)
    videos = os.listdir(dataset)
    print("len folder: ", len(videos))
    for video in tqdm(videos):
        video_path = os.path.join(dataset, video)
        print("video_path: ",video_path)
        cap = cv2.VideoCapture(video_path)
        fas_real = 0
        fas_fake = 0
        count_nonface = 0
        count_frame = 0
        prediction = ''
        while cap.isOpened():
          
            cap.set(cv2.CAP_PROP_POS_FRAMES,25) 
            ret, frame = cap.read()

            if ret:
                if non_face > 3:
                    break
                if count_frame == 5:
                    if fas_real > fas_fake:
                        prediction = 'real'
                    else:
                        prediction = 'fake'
                    break
                        
                if lowlight_enhancer.is_lowlight(frame,threshold):
                    frame = lowlight_enhancer.enhance(frame)
                    cv2.imshow('video',frame)
                    print("yes")
                if fas_model1.is_face(frame) is None or fas_model2.is_face(frame) is None:
                    non_face += 1
                    print("None-face: ",non_face)
                    continue
                else: 
                    count_frame += 1
                    print("fas", count_frame)
                    predict = fas_model1.predict(frame) + fas_model2.predict(frame)
                    output = np.argmax(predict)
                    if output == 1:
                        fas_real += 1
                    else:
                        fas_fake += 1
                
            if not ret:
                break
             # Press 'q' to stop the video display
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
            
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
    print("Can't detect face in video: ", non_face)
