import os
import cv2
import warnings
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys

sys.path.append("miniFAS")
from function_model.llie import LowLightEnhancer
from function_model.fas import FaceAntiSpoofing

warnings.filterwarnings("ignore")

dataset = "miniFAS/datasets/Test/anti_spoofing_dark_dataset"
model_1 = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
threshold = 100
scale_factor = 12
model_llie = "miniFAS/model_onnx/ZeroDCE++.onnx"
# save_path_llie = "miniFAS/datasets/Test/anti_spoofing_dark_dataset_enhancement"
# none_face_path = "miniFAS/datasets/Test/anti_spoofing_dark_dataset_non_llie/non_face"
# false_neg_path = (
#     "miniFAS/datasets/Test/anti_spoofing_dark_dataset_non_llie/false_neg"
# )
# false_pos_path = (
#     "miniFAS/datasets/Test/anti_spoofing_dark_dataset_non_llie/false_pos"
# )
false_save = 'miniFAS/datasets/Test/fail_image'
if __name__ == "__main__":
    #
    list_id = []
    list_name_image = []
    list_face_detected = []
    list_enhance = []
    list_grouthtruth = []
    list_predict = []
    list_evaluate = []
    col_id = "ID"
    col_image = "Image"
    col_face_detect = "Face_detected"
    col_enhance = "Light_enhanced"
    col_groundtruth = "Groundtruth"
    col_predicted = "Predicted"
    col_evaluate = "Evaluate"
    #
    
    
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    count_img = 0
    count_none_face = 0
    count_llie = 0
    fas_model1 = FaceAntiSpoofing(model_1)
    fas_model2 = FaceAntiSpoofing(model_2)
    lowlight_enhancer = LowLightEnhancer(scale_factor=12, model_onnx=model_llie)
    dir_label = os.listdir(dataset)
    for label in dir_label:
        # save_path_llie_result = os.path.join(save_path_llie, label)
        # save_non_face_path = os.path.join(none_face_path, label)
        dir_img = os.path.join(dataset, label)
        images = os.listdir(dir_img)
        print("len folder " + label + ": ", len(images))
        TIME_START = time.time()
        for image in tqdm(images):
            #
            count_img += 1
            list_id.append(count_img)
            list_name_image.append(image)
            #
            prediction = np.zeros((1, 3))
            img_path = os.path.join(dir_img, image)
            img = cv2.imread(img_path)  # BGR

            if lowlight_enhancer.is_lowlight(img, threshold):
                img = lowlight_enhancer.enhance(img[:, :, ::-1])  # RGB
                # result_path = os.path.join(save_path_llie_result, image)
                img = img[:, :, ::-1]
                # cv2.imwrite(result_path, img)
                count_llie += 1
                #
            list_enhance.append(0)

            if fas_model1.is_face(img) is None:
                count_none_face += 1
                # non_face_pth = os.path.join(save_non_face_path, image)
                # cv2.imwrite(non_face_pth, img)
                fail_path = os.path.join(false_save, image)
                cv2.imwrite(fail_path, img)

                #
                list_face_detected.append(0)
                list_grouthtruth.append('none')
                list_predict.append('none')
                list_evaluate.append('none')
                continue
            else:
                #
                list_face_detected.append(1)
                #
                prediction = fas_model1.predict(img) + fas_model2.predict(img)
                output = np.argmax(prediction)

                if output != 1 and label == "fake":
                    tp += 1
                    list_grouthtruth.append("fake")
                    list_predict.append("fake")
                    list_evaluate.append("TRUE")
                elif output == 1 and label == "fake":
                    fn += 1
                    # fn_path = os.path.join(false_neg_path, image)
                    # cv2.imwrite(fn_path, img)
                    list_grouthtruth.append("fake")
                    list_predict.append("real")
                    list_evaluate.append("FALSE")
                    fn_path = os.path.join(false_save, image)
                    cv2.imwrite(fn_path, img)
                elif output != 1 and label == "real":
                    fp += 1
                    # fp_path = os.path.join(false_pos_path, image)
                    # cv2.imwrite(fp_path, img)
                    list_grouthtruth.append("real")
                    list_predict.append("fake")
                    list_evaluate.append("FALSE")
                    fp_path = os.path.join(false_save, image)
                    cv2.imwrite(fp_path, img)
                elif output == 1 and label == "real":
                    tn += 1
                    list_grouthtruth.append("real")
                    list_predict.append("real")
                    list_evaluate.append("TRUE")

   
    print("tp:", tp)
    print("fp:", fp)
    print("fn:", fn)
    print("tn:", tn)
    print("count low light: ", count_llie)
    print("count none face: ", count_none_face)
    print("time: ", time.time() - TIME_START)
    # data = pd.DataFrame(
    #     {
    #         col_id: list_id,
    #         col_image: list_name_image,
    #         col_face_detect: list_face_detected,
    #         col_enhance: list_enhance,
    #         col_groundtruth: list_grouthtruth,
    #         col_predicted: list_predict,
    #         col_evaluate: list_evaluate,
    #     }
    # )
    # data.to_excel("Report_non_LLIE_and_FAS.xlsx", sheet_name="sheet1", index=False)