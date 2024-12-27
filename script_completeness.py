import argparse
import os
import numpy as np
import json
from tqdm import tqdm
import gc

results_dir = os.path.join(os.getcwd(), "results_completeness")
os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--method', type=str, required=True, help='method name')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--n_steps', type=int, default=128, help='number of steps')

    opt = parser.parse_args()
    print(f"Computing Axiom of Completeness metric for {opt.model} using {opt.method} method")

    with open(os.path.join(opt.dataroot, f"{opt.model}_predictions.json"), 'r') as file:
        class_file_dict = json.load(file)

    if opt.method in ["ig", "blurig", "gig", "newblurig", "newgig", "newig"]:
        compute_at = [opt.n_steps]
    else:
        raise ValueError(f"Unknown method: {opt.method}")

    sum_absolute_differences = 0
    sum_absolute_output_diff = 0

    for class_idx in tqdm(range(1000)):
        print(f"Processing class {class_idx}")

        for n_step in compute_at:
            saliencies = np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_ig", f"class_{class_idx:03d}_steps_{n_step}.npy"))
            output_diff = np.load(os.path.join(opt.dataroot, "saliencies", opt.model, f"{opt.method}_outputs", f"class_{class_idx:03d}_steps_{n_step}.npy"))

            saliency_sum = np.sum(saliencies, axis=(1, 2))
            absolute_output_diff = np.abs(output_diff)
            
            # Calculate absolute difference for each image
            absolute_differences = np.abs(saliency_sum - absolute_output_diff)
            
            sum_absolute_differences += np.sum(absolute_differences)
            sum_absolute_output_diff += np.sum(absolute_output_diff)

        gc.collect()

    completeness_metric = sum_absolute_differences / sum_absolute_output_diff

    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    
    with open(os.path.join(results_dir, "metrics", f"completeness_{opt.model}_{opt.method}_{opt.n_steps}.txt"), 'w') as file:
        file.write(f"COMPLETENESS METRIC: {completeness_metric:.6f}\n")

    print(f"Results saved in {os.path.join(results_dir, 'metrics')}")
    print(f"Completeness Metric: {completeness_metric:.6f}")
