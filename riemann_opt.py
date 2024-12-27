import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
import math
import json
from utils import load_model, load_transform
from torch.utils.data import DataLoader

import gc
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import gc
from tqdm import tqdm
import os
from utils import load_transform, load_model, CustomDataset, CustomBatchSampler

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class IntegratedGradients:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

        # dimensions dont matter since it will be resized anyway
        BLACK  = Image.fromarray(np.zeros((299, 299, 3), dtype=np.uint8))
        self.BLACK = transform(BLACK).unsqueeze(0) # adds extra first dimension 

    def compute_outputs_gradients(self, input_tensor, indices, batchSize=64):
        gradients = []
        outputs = []

        for i in range(0, input_tensor.shape[0], batchSize):
            # Get the current batch
            batch = input_tensor[i:i+batchSize].to(device)
            batch.requires_grad_()

            current_batchSize = len(batch)
            output = torch.nn.Softmax(dim=1)(self.model(batch))[torch.arange(current_batchSize), indices[i:i+current_batchSize]]

            # Select the outputs at the given indices
            temp = output.sum()
            temp.backward()
            grad = batch.grad.detach().cpu()
            gradients.append(grad)
            batch.grad.zero_()

            outputs.append(output.detach().cpu())

            torch.cuda.empty_cache()
            gc.collect()

        return torch.cat(outputs), torch.cat(gradients, axis=0)

    def straight_path_images(self, images, n_steps):
        x_diff = images - self.BLACK
        path_images = []

        for alpha in np.linspace(0, 1, n_steps):
            x_step = self.BLACK + alpha * x_diff
            path_images.append(x_step)

        path_images = torch.stack(path_images).transpose(0, 1)

        # returns x sequence
        return path_images

    def saliency(self, images, class_idxs, n_steps, compute_at=None,compute_batchSize=128):

        if compute_at is None:
            compute_at = [n_steps]

        sequence = self.straight_path_images(images, n_steps)

        image_shape = sequence.shape[2:]
        batchSize = sequence.shape[0]

        classes = np.repeat(class_idxs, n_steps)

        reshaped_sequence = sequence.reshape((batchSize*n_steps,)+image_shape).detach()

        outputs, gradients = self.compute_outputs_gradients(reshaped_sequence, classes, batchSize=compute_batchSize)

        gradients = gradients.view(*sequence.shape)

        sequence = sequence.detach()

        magnitude = ((gradients[:,1:]-gradients[:,:-1]) * (sequence[:,1:]-sequence[:,:-1]))
        magnitude = torch.abs(magnitude).sum((2,3,4)) * n_steps**2 / sequence[0, 0].numel()
    
        del reshaped_sequence, sequence, outputs
        torch.cuda.empty_cache()
        gc.collect()

        return magnitude[:, :-1]


class BlurIG:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.blurrers = dict()

    def gaussian_blur(self, images, sigma):
        if sigma == 0:
            return images.cpu()

        size = float(min(2*torch.round(4*sigma)+1, 101))
        if sigma not in self.blurrers:
            self.blurrers[sigma] = transforms.GaussianBlur(size, float(sigma)).to(device)
        return self.blurrers[sigma].forward(images).cpu()
        
    def compute_outputs_gradients(self, input_tensor, indices, batchSize=64):
        gradients = []
        outputs = []

        for i in range(0, input_tensor.shape[0], batchSize):
            # Get the current batch
            batch = input_tensor[i:i+batchSize].to(device)
            batch.requires_grad_()

            current_batchSize = len(batch)
            output = torch.nn.Softmax(dim=1)(self.model(batch))[torch.arange(current_batchSize), indices[i:i+current_batchSize]]

            # Select the outputs at the given indices
            temp = output.sum()
            temp.backward()
            grad = batch.grad.detach().cpu()
            gradients.append(grad)
            batch.grad.zero_()

            outputs.append(output.detach().cpu())

            torch.cuda.empty_cache()
            gc.collect()

        return torch.cat(outputs), torch.cat(gradients, axis=0)
    
    def saliency(self, images, prediction_class, steps=20, steps_at=None, batch_size=32, max_sigma = 50, grad_step=0.01, sqrt=False):
        if sqrt:
            sigmas = torch.Tensor([math.sqrt(float(i)*max_sigma/float(steps)) for i in range(0, steps+1)])
        else:
            sigmas = torch.Tensor([float(i)*max_sigma/float(steps) for i in range(0, steps+1)])
        step_vector_diff = sigmas[1:] - sigmas[:-1]

        if steps_at is None:
            steps_at = [steps]

        sequence           = torch.zeros((len(images), steps, *images.shape[1:]))
        gaussian_gradients = torch.zeros((len(images), steps, *images.shape[1:]))

        images = images.to(device)

        for i in range(steps):
            x_step = self.gaussian_blur(images, sigmas[i])
            gaussian_gradient = (self.gaussian_blur(images, sigmas[i]+grad_step)-x_step)/grad_step

            gc.collect()
            torch.cuda.empty_cache()

            gaussian_gradients[:, i] = gaussian_gradient
            sequence[:, i] = x_step

        images = images.cpu()

        target_class_idx = np.repeat(prediction_class, steps)
        outputs, gradients = self.compute_outputs_gradients(sequence.view(-1, *images.shape[1:]), target_class_idx, batchSize=batch_size)
        outputs = outputs.view(*sequence.shape[:2])
        gradients = gradients.view(*sequence.shape)


        for n in steps_at:
            gradients_copy = gradients[:,::steps//n]
            gaussian_gradients_copy = gaussian_gradients[:,::steps//n]
            step_vector_diff_copy = step_vector_diff.reshape(n, steps//n).sum(1).view(1,-1,1,1,1)

            magnitude = gaussian_gradients_copy*gradients_copy
            magnitude = (magnitude[:, 1:]-magnitude[:, :-1]) / step_vector_diff_copy[:, :-1]
            magnitude = torch.abs(magnitude).sum((2,3,4)) / sequence[0, 0].numel()

        return torch.flip(magnitude, (1,))


epsilon = 1E-7

def l1_distance(x1, x2):
  return np.abs(x1 - x2).sum()

def translate_x_to_alpha(x, x_input, x_baseline):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(x_input - x_baseline != 0,
                    (x - x_baseline) / (x_input - x_baseline), np.nan)

def translate_alpha_to_x(alpha, x_input, x_baseline):
  assert 0 <= alpha <= 1.0
  return x_baseline + (x_input - x_baseline) * alpha

class GIG:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform

    def compute_outputs_gradients(self, input_tensor, indices, batchSize=64):
        gradients = []
        outputs = []

        for i in range(0, input_tensor.shape[0], batchSize):
            # Get the current batch
            batch = input_tensor[i:i+batchSize].to(device)
            batch.requires_grad_()

            current_batchSize = len(batch)
            output = torch.nn.Softmax(dim=1)(self.model(batch))[torch.arange(current_batchSize), indices[i:i+current_batchSize]]

            # Select the outputs at the given indices
            temp = output.sum()
            temp.backward()
            grad = batch.grad.detach().cpu()
            gradients.append(grad)
            batch.grad.zero_()

            outputs.append(output.detach().cpu())

            torch.cuda.empty_cache()
            gc.collect()

        return torch.cat(outputs), torch.cat(gradients, axis=0)


    def saliency(self, x_images, class_idxs, steps=25, compute_at=[], batchSize=64, fraction=0.50, max_dist=0.02):

        x_orig = x_images.numpy()
        x_input = x_orig.astype(np.float32)

        x_input_all = np.asarray(x_input, dtype=np.float32)
        x_baseline = np.zeros(x_input.shape, dtype=np.float32)
        x_all = x_baseline.copy()
        l1_total_all = np.asarray([l1_distance(x_input, x_baseline[0]) for x_input in x_input_all])
        attr_all = np.zeros_like(x_input, dtype=np.float32)
        integrand_all = np.zeros_like(x_input, dtype=np.float32)[:, None].repeat(steps, axis=1)

        for step in range(steps):
            x_preprocessed_all = torch.from_numpy(x_all)
            outputs_all, grad_actual_all = self.compute_outputs_gradients(x_preprocessed_all, class_idxs, batchSize=batchSize)
            outputs_all, grad_actual_all = outputs_all.numpy(), grad_actual_all.numpy()
            grad_all = grad_actual_all.copy()
            alpha = (step + 1.0) / steps
            alpha_min = max(alpha - max_dist, 0.0)
            alpha_max = min(alpha + max_dist, 1.0)

            for i in range(len(x_all)):
                gamma = np.inf
                x = x_all[i]
                x_input = x_input_all[i]
                grad_actual = grad_actual_all[i]
                grad = grad_all[i]
                x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline[0])
                x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline[0])
                l1_target = l1_total_all[i] * (1 - (step + 1) / steps)

                while gamma > 1.0:
                    x_old = x.copy()
                    x_alpha = translate_x_to_alpha(x, x_input, x_baseline[0])
                    x_alpha[np.isnan(x_alpha)] = alpha_max
                    x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

                    l1_current = l1_distance(x, x_input)
                    if math.isclose(l1_target, l1_current, rel_tol=epsilon, abs_tol=epsilon):
                        integrand_all[i, step] = (x - x_old) * grad_actual
                        attr_all[i] += (x - x_old) * grad_actual
                        break

                    grad[x == x_max] = np.inf

                    threshold = np.quantile(np.abs(grad), fraction, method='lower')
                    s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)

                    l1_s = (np.abs(x - x_max) * s).sum()
                    if l1_s > 0:
                        gamma = (l1_current - l1_target) / l1_s
                    else:
                        gamma = np.inf

                    if gamma > 1.0:
                        x[s] = x_max[s]
                    else:
                        assert gamma > 0, gamma
                        x[s] = translate_alpha_to_x(gamma, x_max, x)[s]

                    integrand_all[i, step] = (x - x_old) * grad_actual
                    attr_all[i] += (x - x_old) * grad_actual

        return torch.from_numpy(np.abs(integrand_all[: , 1:] - integrand_all[:, :-1]).sum((2,3,4)) * steps**2)
    

model, transform = load_model("inceptionv3"), load_transform("inceptionv3")
integrator = BlurIG(model, transform)

with open(os.path.join("/scratch/shree_s.iitr/imagenet", f"inceptionv3_predictions.json"), 'r') as file:
    class_file_dict = json.load(file) # not really a dict, its a list

CLASS_CLUSTER = 4

image_pathss = []
indexss = []
for i in range(0, 1000, CLASS_CLUSTER):
    indexs = []
    image_paths=[]
    for class_idx in range(i, i+CLASS_CLUSTER):
        image_paths.extend([os.path.join("/scratch/shree_s.iitr/imagenet", "val/images", file ) for file in class_file_dict[class_idx]])
        indexs += [class_idx] * len(class_file_dict[class_idx])
    image_pathss.append(image_paths)
    indexss.append(indexs)
    
dataset = CustomDataset(transform=transform)
batch_sampler = CustomBatchSampler(image_pathss, indexss)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

outputs = []
for i, (images, indexs) in tqdm(zip(range(0, 1000, CLASS_CLUSTER), dataloader), total=1000//CLASS_CLUSTER):
    print("Processing classes", i, "to", i+CLASS_CLUSTER)
    
    output = integrator.saliency(images, indexs, 128, None, 128)
    outputs.append(output)

outputs = torch.vstack(outputs).numpy()
np.save("magnitude_blurig", outputs)