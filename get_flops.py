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

def parse_model_names(model_list_file):
    model_names = []
    with open(model_list_file) as f:
        for line in f:
            if line.strip() == "": continue
            model_names.append(line.strip())
    return model_names


def get_mac_flops(model_names, input_h, input_w, batch_size, model_source):
    timm_models = timm.list_models(pretrained=False)

    flops_params = {}
    input = torch.randn(batch_size, 3, input_h, input_w)
    for model_name in tqdm(model_names):
        try:
            if model_source == "timm":
                model = timm.create_model(model_name, pretrained=False)
            else:
                model = eval(model_name)(pretrained=False)

            macs, params = profile(model, inputs=(input, ), verbose=False)
            flops_params[model_name] = (macs, params)
        except Exception as e:
            flops_params[model_name] = (0, 0)
            print(model_name, e)

    result_dir = "flops"
    if not os.path.exists(result_dir): os.makedirs(result_dir)
    with open(f"{result_dir}/model_mac_flops_{input_h}x{input_w}_{model_source}.csv", "a") as f:
        f.write("model,macs(G),params(M)\n")
        for model_name in model_names:
            macs, params = flops_params[model_name]
            f.write(f"{model_name},{macs/(1000**3)},{params/(1000**2)}\n")
            print(f"model={model_name}, macs={macs/(1000**3)}G, params={params/(1000**2)}M")
 

def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <input_h> <input_w> <batch_size>")
        return
    input_h = int(sys.argv[1])
    input_w = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    timm_model_names = parse_model_names("model_names/model_timm.txt")    
    get_mac_flops(timm_model_names, input_h, input_w, batch_size, "timm")

    torch_model_names = parse_model_names("model_names/model_pytorch.txt")    
    get_mac_flops(torch_model_names, input_h, input_w, batch_size, "pytorch")

if __name__ == '__main__':
    main()