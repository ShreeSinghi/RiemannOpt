import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np
import json
from PIL import Image
import gc
from tqdm import tqdm

import torch
from utils import load_transform, load_model, compute_outputs
from metrics import insertion_score
from torch.utils.data import DataLoader, Sampler, Dataset

cudnn.benchmark = True

import psutil

class CustomDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, batch):
        image_path, index = batch
        return self.transform(Image.open(image_path).convert("RGB")), index

class CustomBatchSampler(Sampler):
    def __init__(self, image_pathss, indexss):
        self.image_pathss = image_pathss
        self.indexss = indexss

    def __iter__(self):
        for image_paths, indexs in zip(self.image_pathss, self.indexss):
            yield zip(image_paths, indexs)

    def __len__(self):
        return len(self.img_path_list)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--method', type=str, required=True, help='method name')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--resume_epoch', type=int, default=0, help='input batch size')
parser.add_argument('--n_steps', type=int, default=128, help='number of steps')

opt = parser.parse_args()
batchSize = opt.batchSize
print(f"Computing insertion score for {opt.model} using {opt.method} method")

model = load_model(opt.model)
transform = load_transform(opt.model)

BLACK = -torch.Tensor((0.485, 0.456, 0.406))/torch.Tensor((0.229, 0.224, 0.225))

with open(os.path.join(opt.dataroot, f"{opt.model}_predictions.json"), 'r') as file:
    class_file_dict = json.load(file) # not really  a dict, its a list

if opt.method == "ig":
    compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
if opt.method == "blurig":
    compute_at = [2**i for i in range(opt.n_steps+1) if  8 <= 2**i <= opt.n_steps]
if opt.method == "gig":
    compute_at = [opt.n_steps]
if opt.method == "newblurig":
    compute_at = [opt.n_steps]
if opt.method == "newig":
    compute_at = [opt.n_steps]
if opt.method == "newgig":
    compute_at = [opt.n_steps]

insertion_scoress_ig = []

insertion_scoress_ig_norm = []

for class_idx in tqdm(range(opt.resume_epoch, 1000)):
    print("Processing class", class_idx)

    images = [Image.open(os.path.join(opt.dataroot, r"val/images", file )).convert('RGB') for file in class_file_dict[class_idx]]
    
    preprocessed_images = torch.stack([transform(image) for image in images])

    saliencies_ig   = torch.from_numpy(np.stack([np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}.npy")) for n_step in compute_at])).abs()
    
    indexs = torch.Tensor([class_idx] * (len(class_file_dict[class_idx]) * len(compute_at))).long()
    blacks = (torch.ones_like(preprocessed_images) * BLACK.view(1, 3, 1, 1))
    
    insertion_scores_ig = insertion_score(model, saliencies_ig.view(-1, *saliencies_ig.shape[2:]),
                                            preprocessed_images.repeat(len(compute_at), 1, 1, 1),
                                            blacks.repeat(len(compute_at),1,1,1),
                                            indexs,
                                            batchSize=batchSize)


    insertion_scoress_ig_norm.append(insertion_scores_ig[0].reshape(len(compute_at), len(class_file_dict[class_idx])))

    insertion_scoress_ig.append(insertion_scores_ig[1].reshape(len(compute_at), len(class_file_dict[class_idx])))

    gc.collect()

insertion_scoress_ig = np.concatenate(insertion_scoress_ig, axis=1)
insertion_scoress_ig_norm = np.concatenate(insertion_scoress_ig_norm, axis=1)

os.makedirs("results", exist_ok=True)
with open(f"results/insertion_{opt.model}_{opt.method}_ig_{opt.n_steps}.txt", 'w') as file:
    file.write(f"IG MEAN INSERTION SCORE {insertion_scoress_ig.mean(axis=1)}\n")
    file.write(f"IG MEAN NORMALIZED INSERTION SCORE {insertion_scoress_ig_norm.mean(axis=1)}\n")
    file.write(str(compute_at))