import argparse
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
import json
from tqdm import tqdm
from utils import load_transform, load_model

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_filename_class_dict():
    filename_class_dict = {}
    annotation_file_path = os.path.join(opt.dataroot, r"val/val_annotations.txt")
    with open(annotation_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            filename = parts[0]
            class_name = parts[1]
            filename_class_dict[filename] = class_name
    
    return filename_class_dict

def create_index_filename_list(filename_class_dict):
    filename_index_dict = {}
    wnids_file_path = os.path.join(opt.dataroot, r"wnids.txt")
    
    with open(wnids_file_path, 'r') as file:
        labels = file.read().strip().splitlines()
    
    class_name_to_index = {label: index for index, label in enumerate(labels)}
    
    for filename, class_name in filename_class_dict.items():
        if class_name in class_name_to_index:
            filename_index_dict[filename] = class_name_to_index[class_name]

    index_file_list = [[] for i in range(1000)]

    for file_name, class_index in filename_index_dict.items():
        index_file_list[class_index].append(file_name)

    return index_file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')

    opt = parser.parse_args()
    batchSize = opt.batchSize

    cudnn.benchmark = True

    original_files = os.listdir(os.path.join(opt.dataroot, r"val/images"))
    files = [os.path.join(opt.dataroot, r"val/images", file) for file in original_files]
    
    transform = load_transform(opt.model)
    model = load_model(opt.model)

    predictions = []
    probas = []

    filename_class_dict = create_filename_class_dict()
    index_filename_list = create_index_filename_list(filename_class_dict)

    print("Predicting classes...")
    with torch.no_grad():
        for batch in tqdm([files[i:i + batchSize] for i in range(0, len(files), batchSize)]):
            batch = transform(batch)
            batch = batch.to(device)
            result = torch.nn.functional.softmax(model(batch))
            indexes = result.argmax(1).cpu()
            probas.extend(result.cpu()[np.arange(len(batch)), indexes].tolist())
            predictions.extend(indexes.tolist())
            del batch, result

    predictions_index_dict = dict(zip(original_files, list(zip(predictions, probas))))
    filename_class_dict = create_filename_class_dict()
    index_filename_list = create_index_filename_list(filename_class_dict)
    

    index_file_list = [[] for i in range(1000)]

    for file_name, (class_index, proba) in tqdm(predictions_index_dict.items()):
        if file_name in index_filename_list[class_index] and proba > 0.8:
            index_file_list[class_index].append(file_name)

    with open(os.path.join(opt.dataroot, f'{opt.model}_probas.json'), 'w') as file:
        json.dump(index_file_list, file, indent=4)