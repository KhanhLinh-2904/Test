import cv2
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import sys
sys.path.append('miniFAS')
from function_model.llie import LowLightEnhancer
from function_model.fas import FaceAntiSpoofing
from utils.custom_utils import detect_face, tracking


model_1 = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
model_llie = 'miniFAS/model_onnx/ZeroDCE++.onnx'
threshold = 100
scale_factor = 12
fas_model1 = FaceAntiSpoofing(model_1)
fas_model2 = FaceAntiSpoofing(model_2)
lowlight_enhancer = LowLightEnhancer(scale_factor=12, model_onnx=model_llie)
def camera(frame_fas, result_fas, frame_verify, result_verify):
    batch_face = []
    start_fas = False
    label, color = False, (0, 0, 0)

   
    # Create a VideoCapture object called cap
    cap = cv2.VideoCapture(0)

    # This is an infinite loop that will continue to run until the user presses the `q` key
    count_frame = 0
    while cap.isOpened():
        tic = time.time()

        # Read a frame from the webcam
        ret, frame_root = cap.read()
        # frame_number = 25
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        frame_root = cv2.flip(frame_root, 1)
        frame = frame_root.copy()

        # If the frame was not successfully captured, break out of the loop
        if ret is False:
            break
        if lowlight_enhancer.is_lowlight(frame,threshold):
            frame = lowlight_enhancer.enhance(frame)
        image_bbox,_ = detect_face(frame)

        if start_fas and image_bbox is not None:
            batch_face.append((image_bbox, frame))
            count_frame += 1
        
        if count_frame == 5:
            frame_fas.put(batch_face)
            # print("put")
            count_frame = 0
            batch_face = []
            start_fas = False

        if not result_fas.empty():
            label = result_fas.get()

        new_gister = tracking(image_bbox, frame)
        if new_gister:
            start_fas = True
        
        test_speed = time.time() - tic
        fps = 1/test_speed

        if label:
            if label == "REAL":
                color = (0, 255, 0)
            elif label == "FAKE":
                color = (0, 0, 255)
        
        
        cv2.rectangle(frame_root, (image_bbox[0], image_bbox[1]), (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)

        # Display the FPS on the frame
        cv2.putText(frame_root, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame on the screen
        cv2.imshow("frame", frame_root)

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
        else:
            result_queue.put("FAKE")

if __name__ == "__main__":
    frame_verify = multiprocessing.Queue()
    result_verify = multiprocessing.Queue()
    frame_fas = multiprocessing.Queue()
    result_fas = multiprocessing.Queue()

    p1 = multiprocessing.Process(name='p1', target=camera, args=(frame_fas, result_fas, frame_verify, result_verify))
    p = multiprocessing.Process(name='p', target=anti_spoofing, args=(frame_fas, result_fas))
    p.start()
    p1.start()
    p.join()
    p1.join()
    
