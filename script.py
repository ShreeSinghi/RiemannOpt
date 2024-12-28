import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
import json
from PIL import Image
import gc
from tqdm import tqdm

from torch.utils.data import DataLoader

from ig import IntegratedGradients
from blurig import BlurIG
from gig import GIG
from newblurig import NewBlurIG
from newgig import NewGIG
from newig import NewIG
from utils import load_transform, load_model, CustomDataset, CustomBatchSampler
cudnn.benchmark = True
      
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--method', type=str, required=True, help='method name')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--resume_epoch', type=int, default=0, help='input batch size')
parser.add_argument('--n_steps', type=int, default=128, help='number of steps')
parser.add_argument('--class_cluster', type=int, default=1, help='number of classes to compute at one go')

opt = parser.parse_args()
batchSize = opt.batchSize
CLASS_CLUSTER = opt.class_cluster
print(f"Computing saliencies for {opt.model} using {opt.method} method")


model = load_model(opt.model)
transform = load_transform(opt.model)

with open(os.path.join(opt.dataroot, f"{opt.model}_predictions.json"), 'r') as file:
    class_file_dict = json.load(file) # not really  a dict, its a list

if opt.method == "ig":
    integrator = IntegratedGradients(model, transform)
    compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
if opt.method == "newig":
    integrator = NewIG(model, transform, n_points=opt.n_steps)
    compute_at = [opt.n_steps]
if opt.method == "blurig":
    integrator = BlurIG(model, transform)
    compute_at = [opt.n_steps]
if opt.method == "gig":
    integrator = GIG(model, transform)
    compute_at = [opt.n_steps]
if opt.method == "newblurig":
    integrator = NewBlurIG(model, transform, n_points=opt.n_steps)
    compute_at = [opt.n_steps]
if opt.method == "newgig":
    integrator = NewGIG(model, transform, n_points=opt.n_steps)
    compute_at = [opt.n_steps]
    
image_pathss = []
indexss = []
for i in range(opt.resume_epoch, 1000, CLASS_CLUSTER):
    indexs = []
    image_paths=[]
    for class_idx in range(i, i+CLASS_CLUSTER):
        image_paths.extend([os.path.join(opt.dataroot, "val/images", file ) for file in class_file_dict[class_idx]])
        indexs += [class_idx] * len(class_file_dict[class_idx])
    
    image_pathss.append(image_paths)
    indexss.append(indexs)

dataset = CustomDataset(transform=transform)
batch_sampler = CustomBatchSampler(image_pathss, indexss)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

for i, (images, indexs) in zip(tqdm(range(opt.resume_epoch, 1000, CLASS_CLUSTER)), dataloader):
    print(f"Processing classes {i} to {i+CLASS_CLUSTER-1}")
    
    saliencies_ig, outputs_diff = integrator.saliency(images, indexs, opt.n_steps, compute_at, batchSize)
    pointer = 0
    for class_idx in range(i, i+CLASS_CLUSTER):
        class_size = len(class_file_dict[class_idx])

        for suffix in ['ig', 'outputs']:
            os.makedirs(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_{suffix}"), exist_ok=True)

        for j, n_step in enumerate(compute_at):
            ig_data = saliencies_ig[j][pointer:pointer+class_size]
            output_diff_data = outputs_diff[pointer:pointer+class_size]

            np.save(
                os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}"),
                ig_data.astype(np.float16)
            )
            np.save(
                os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_outputs", f"class_{class_idx:03d}_steps_{n_step}"),
                output_diff_data.astype(np.float16)
            )
        
        pointer += class_size
                
    del saliencies_ig, outputs_diff
    gc.collect()

