import os
import torch
import torch.optim
import sys
sys.path.append('miniFAS')
from src.anti_spoof_predict import AntiSpoofPredict


model_1 = 'miniFAS/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'miniFAS/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model_save = 'miniFAS/model_onnx'

def Convert_ONNX(model_path, model_save):
    torch_model = AntiSpoofPredict(0)
    torch_model_loaded = torch_model._load_model(model_path)
    torch_model_loaded.eval()
    model_name = model_path.split("/")[-1].split(".pth")[0]
    model_name = model_name +'.onnx'
    dummy_input = torch.randn(1, 3, 80, 80, requires_grad=True)
    save_path = os.path.join(model_save, model_name)
    # Export the model
    torch.onnx.export(
        torch_model_loaded,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        save_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
        dynamic_axes={
            "modelInput": {
                0: "batch_size",
                2: "height",
                3: "width",
            },  # variable length axes
            "modelOutput": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print(" ")
    print("Model has been converted to ONNX")

if __name__ == '__main__':
    for model_path in [model_1, model_2]:
        Convert_ONNX(model_path, model_save)