import torch
import torch.optim
import sys
sys.path.append('miniFAS')
from Zero_DCE_plus_plus import model


model_path = 'miniFAS/Zero_DCE_plus_plus/Epoch99.pth'
model_save = 'miniFAS/model_onnx/ZeroDCE++.onnx'
scale_factor = 12

def Convert_ONNX(scale_factor, model_path, save_path):
    torch_model = model.enhance_net_nopool(scale_factor)
    torch_model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    torch_model.eval()

    # Let's create a dummy input tensor
    input_size = (1, 3, 3648, 5472)
    dummy_input = torch.randn(1, 3, 3648, 5472, requires_grad=True)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
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
    Convert_ONNX(scale_factor, model_path, model_save)