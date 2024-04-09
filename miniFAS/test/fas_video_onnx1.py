import cv2
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import os
import sys

from tqdm import tqdm
sys.path.append('miniFAS')
from function_model.llie import LowLightEnhancer
from function_model.fas import FaceAntiSpoofing
from utils.custom_utils import detect_face, tracking


model_1 = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
model_llie = 'miniFAS/model_onnx/ZeroDCE++.onnx'
dataset ="miniFAS/datasets/Test/low-light-video"

threshold = 100
scale_factor = 12
fas_model1 = FaceAntiSpoofing(model_1)
fas_model2 = FaceAntiSpoofing(model_2)
lowlight_enhancer = LowLightEnhancer(scale_factor=12, model_onnx=model_llie)


def camera(frame_fas, video_path):
    batch_face = []
    start_fas = False

    path = video_path.get()
    # Create a VideoCapture object called cap
    cap = cv2.VideoCapture(path)

    # This is an infinite loop that will continue to run until the user presses the `q` key
    count_frame = 0
    while cap.isOpened():
        tic = time.time()

        # Read a frame from the webcam
        ret, frame_root = cap.read()
        frame_root = cv2.flip(frame_root, 1)
        frame = frame_root.copy()

        # If the frame was not successfully captured, break out of the loop
        if ret is False:
            break
        if lowlight_enhancer.is_lowlight(frame,threshold):
            print("LLIE")
            frame = lowlight_enhancer.enhance(frame)
        image_bbox = fas_model1.is_face(frame)
        if image_bbox is None:
            new_gister = False
        else:
            print("trackinggg")
            new_gister = tracking(image_bbox, frame)
        if start_fas and image_bbox is not None:
            batch_face.append((image_bbox, frame))
            count_frame += 1
        
        if count_frame == 5:
            frame_fas.put(batch_face)
            # print("put")
            count_frame = 0
            batch_face = []
            start_fas = False
            print("Stoppp")
            break

        # new_gister = tracking(image_bbox, frame)
        if new_gister:
            start_fas = True
        
        test_speed = time.time() - tic
        fps = 1/test_speed

        # Check if the user has pressed the `q` key, if yes then close the program.
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release the VideoCapture object
    cap.release()

    # Close all open windows
    cv2.destroyAllWindows()

def anti_spoofing(frame_queue, result_queue):
    while True:
        real, fake = 0, 0

        # Get frame from the queue
        detections = frame_queue.get()

        for (bbox, frame) in detections:
            frame = np.asarray(frame, dtype=np.uint8) 
            prediction = fas_model1.predict(frame) + fas_model2.predict(frame)
            output = np.argmax(prediction)

            if output == 1:
                real += 1
            else:
                fake += 1
        
        if real > fake:
            result_queue.put("REAL")
            # print("Yes true!")
        else:
            result_queue.put("FAKE")
def metric(video_path,result_fas):
    label = 'REAL'
    if not result_fas.empty():
        print("Process result")
        output = result_fas.get()
        if output == label:
            print("Yes")
        else:
            print("No")
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    
    videos = os.listdir(dataset)
    print("len folder: ", len(videos))
    for video in tqdm(videos):
        frame_fas = multiprocessing.Queue()
        result_fas = multiprocessing.Queue()
        video_path = multiprocessing.Queue()
        path = os.path.join(dataset, video)
        video_path.put(path)
        print("video_path: ",video_path)
        p1 = multiprocessing.Process(name='p1', target=camera, args=(frame_fas, video_path))
        p2 = multiprocessing.Process(name='p2', target=anti_spoofing, args=(frame_fas, result_fas))
        p3 = multiprocessing.Process(name='p3', target=metric, args=(video_path, result_fas))

        p1.start()
        p2.start()
        p3.start()
        p1.join()
        p2.join()
        p3.join()

        
        # if not result_fas.empty():
        #     print("Process result")
        #     output = result_fas.get()
        #     if output == "FAKE" and label == 'FAKE':
        #         tp += 1
        #     elif output == 'REAL' and label == 'FAKE':
        #         fn += 1
        #     elif output == 'FAKE' and label == 'REAL':
        #         fp += 1
        #     elif output == 'REAL' and label == 'REAL':
        #         tn += 1   
    
