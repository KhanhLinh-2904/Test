import cv2
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('miniFAS')
from function_model.llie import LowLightEnhancer
from function_model.fas import FaceAntiSpoofing
from utils.custom_utils import detect_face, tracking
from tqdm import tqdm

model_1 = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
model_llie = 'miniFAS/model_onnx/ZeroDCE++.onnx'
dataset = 'miniFAS/datasets/Test/low-light-video'

threshold = 100
scale_factor = 12
fas_model1 = FaceAntiSpoofing(model_1)
fas_model2 = FaceAntiSpoofing(model_2)
lowlight_enhancer = LowLightEnhancer(scale_factor=12, model_onnx=model_llie)
def camera(video_path):
    frame_fas = []
    # Create a VideoCapture object called cap
    cap = cv2.VideoCapture(video_path)
    # frame_number = 25
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    # This is an infinite loop that will continue to run until the user presses the `q` key
    count_frame = 0
    count_non_face = 0
    num_frames = 0
    while cap.isOpened():
       
        ret, frame = cap.read()
        num_frames += 1
        if ret is False:
            break
        if lowlight_enhancer.is_lowlight(frame,threshold):
            frame = lowlight_enhancer.enhance(frame)
        image_bbox = fas_model1.is_face(frame)
        if image_bbox is not None:
            new_gister = tracking(image_bbox, frame)
        else:
            new_gister = False
            count_non_face += 1
        if new_gister and image_bbox is not None:
            frame_fas.append(frame)
            count_frame += 1
        
        if count_frame == 9:
            break
            
        # Check if the user has pressed the `q` key, if yes then close the program.
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release the VideoCapture object
    cap.release()

    return frame_fas, num_frames

def anti_spoofing(frame_fas):
    while True:
        real, fake = 0, 0

        # Get frame from the queue
        detections = frame_fas

        for frame in detections:
            frame = np.asarray(frame, dtype=np.uint8) 
            prediction = fas_model1.predict(frame) + fas_model2.predict(frame)
            output = np.argmax(prediction)

            if output == 1:
                real += 1
            else:
                fake += 1
        
        if real > fake:
           return 'REAL'
        else:
            return 'FAKE'

    
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    label = 'REAL'
    total_frames = 0
    
    videos = os.listdir(dataset)
    print("len folder: ", len(videos))
    # for video in tqdm(videos):
    start_time = time.time()
    for video in videos:
        video_path = os.path.join(dataset, video)
        print("video_path: ",video_path)
       
        frame_fas, num_frames = camera(video_path)
        total_frames += num_frames
        output = anti_spoofing(frame_fas)
        print('output: ', output)
        print('--------------------------------------------------------------------------------------')

        if output == "FAKE" and label == 'FAKE':
            tp += 1
        elif output == 'REAL' and label == 'FAKE':
            fn += 1
        elif output == 'FAKE' and label == 'REAL':
            fp += 1
        elif output == 'REAL' and label == 'REAL':
            tn += 1  
    total_time = time.time() - start_time
    
    print('--------------------------------------------------------------------------------------')
    print('fusion matrix')
    print('tp: ', tp)
    print('tn: ', tn)
    print('fp: ', fp)
    print('fn: ', fn)
    print('Total time: ', total_time)
    print('Total frames: ', total_frames)
