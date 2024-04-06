import cv2
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os
import torch.optim
import numpy as np
import torchvision 
import cv2
from matplotlib import pyplot as plt
from torchvision import  utils
import Zero_DCE_plus_plus.model as model
model_path = 'Zero_DCE_plus_plus/Epoch99.pth'
model_save = 'model_onnx/ZeroDCE++1.onnx'
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def is_low_light(image, threshold=50):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_intensity = cv2.mean(gray_image)[0]
    if average_intensity < threshold:
        return True
    else:
        return False

def lowlight_enhancement_onnx(data_lowlight, image_name, save_model=False, save_path="./datasets/Test/"):
    scale_factor = 12
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0)
    
    torch_model = model.enhance_net_nopool(scale_factor)
    torch_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    torch_model.eval()


    onnx_program = torch.onnx.dynamo_export(torch_model, data_lowlight)
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(data_lowlight)
    
    onnx_program.save(model_save)

    ort_session = onnxruntime.InferenceSession(model_save, providers=['CPUExecutionProvider'])
    
    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
   
    output1 = onnxruntime_outputs[0].reshape(onnxruntime_outputs[0].shape[1], onnxruntime_outputs[0].shape[2], onnxruntime_outputs[0].shape[3])

    red_channel = output1[0]
    green_channel = output1[1]
    blue_channel = output1[2]
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    rgb_image = torch.from_numpy(rgb_image)
    grid = torchvision.utils.make_grid(rgb_image)
    rgb_image = grid.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()

    if save_model:
        result_path = os.path.join(save_path, image_name)
        plt.imsave(result_path, rgb_image)
    return rgb_image

def lowlight_enhancement_pytorch(data_lowlight,image_name, save_model=False, save_path='./datasets/Test/'):
    scale_factor = 12
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0)
	
    torch_model = model.enhance_net_nopool(scale_factor)
    torch_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    torch_model.eval()

    enhanced_image, para = torch_model(data_lowlight)
    
    grid = torchvision.utils.make_grid(enhanced_image)
    enhanced_image = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    if save_model:
        result_path = os.path.join(save_path, image_name)
        plt.imsave(result_path, enhanced_image)
    return enhanced_image