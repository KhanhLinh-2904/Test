import os
import cv2
import warnings
from tqdm import tqdm
import torch
from miniFAS.function_model.llie import is_low_light, lowlight_enhancement_onnx
from miniFAS.function_model.miniFAS import miniFAS_onnx
warnings.filterwarnings('ignore')
# Initialize GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset ="datasets/Test/low-light-video"

if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    label = 'real'
    threshold = 100
    non_face = 0
    videos = os.listdir(dataset)
    for video in tqdm(videos):
        video_path = os.path.join(dataset, video)
        print("video_path: ",video_path)
        cap = cv2.VideoCapture(video_path)
        fas_false = 0
        fas_true = 0
        count_nonface = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES,25) 
            ret, frame = cap.read()
            if ret:
                if is_low_light(frame,threshold):
                    frame = lowlight_enhancement_onnx(frame)
                prediction = miniFAS_onnx(frame)
                if count_nonface >= 10:
                    non_face += 1
                    break
                if prediction == "none":
                    count_nonface += 1
                    continue
                else:
                    break
                
            else:
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
