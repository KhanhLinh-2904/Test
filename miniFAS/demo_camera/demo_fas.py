import cv2
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from miniFAS.function_model.fas import miniFAS_onnx
from miniFAS.function_model.llie import is_low_light, lowlight_enhancement_onnx
from utils.custom_utils import detect_face, predict_fas, tracking

threshold = 100
model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'


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
        frame_root = cv2.flip(frame_root, 1)
        frame = frame_root.copy()

        # If the frame was not successfully captured, break out of the loop
        if ret is False:
            break
        if is_low_light(frame,threshold):
            frame = lowlight_enhancement_onnx(frame)
        image_bbox, _ = detect_face(frame)

        if start_fas:
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
            
            label = miniFAS_onnx(frame)

            if label == "real":
                real += 1
            elif label == "fake":
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
    
