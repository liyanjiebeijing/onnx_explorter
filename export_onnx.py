import torch
import torchvision
from torchvision.models import *
from thop import profile
import torch.nn as nn
from tqdm import tqdm
import json
import os
import sys
import timm
from parser import parse_model_names

def export_onnx(model_names, input_h, input_w, batch_size, model_source = "timm"):
    timm_models = timm.list_models(pretrained=True)    
    with open("all_timm_model.json", "w") as f:
        f.write(json.dumps(timm_models, indent=4))

    onnx_dir = f"onnx_{input_h}x{input_w}_batchsize_{batch_size}"
    if not os.path.exists(f"{onnx_dir}"): os.makedirs(f"{onnx_dir}")
    exist_model_names = [each.replace(".onnx", "") for each in os.listdir(onnx_dir)]

    input = torch.randn(batch_size, 3, input_h, input_w)
    for model_name in tqdm(model_names):
        if model_name in exist_model_names: continue
        try:
            if model_source == "timm":
                model = timm.create_model(model_name, pretrained=True)
            else:
                model = eval(model_name)(pretrained=True)
        except Exception as e:            
            print(e)
            continue

        model.eval()

        try:
            # torch.onnx.export(model,           # model being run
            #         input,                     # model input (or a tuple for multiple inputs)
            #         f"{onnx_dir}/{model_name}.onnx",# where to save the model (can be a file or file-like object)
            #         export_params=True,        # store the trained parameter weights inside the model file
            #         opset_version=13,          # the ONNX version to export the model to
            #         do_constant_folding=True,  # whether to execute constant folding for optimization
            #         input_names = ['input'],   # the model's input names
            #         output_names = ['output'], # the model's output names
            #         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
            #                         'output' : {0 : 'batch_size'}})            
            torch.onnx.export(model,           # model being run
                    input,                     # model input (or a tuple for multiple inputs)
                    f"{onnx_dir}/{model_name}.onnx",# where to save the model (can be a file or file-like object)                    
                    opset_version=11,
                    input_names = ['input'],   # the model's input names
                    output_names = ['output']) # the model's output names)
        except Exception as e:
            print(f"failed to export {model_name} to onnx: {e}")
            

def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <input_h> <input_w> <batch_size>")
        return
    input_h = int(sys.argv[1])
    input_w = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    timm_model_names = parse_model_names("model_names/model_timm.txt")    
    export_onnx(timm_model_names, input_h, input_w, batch_size, "timm")

    timm_model_names = parse_model_names("model_names/model_pytorch.txt")    
    export_onnx(timm_model_names, input_h, input_w, batch_size, "pytorch")

if __name__ == '__main__':
    main()